#!/usr/bin/env python3
"""Create bounded, paginated local comparisons from grouped batch outputs.

Reports are diagnostics, not acceptance evidence. A bad scene gets an error
card; it cannot prevent other scenes from being inspected. Absolute source
photos may live outside the batch, while generated artifacts stay inside their
own scene directory.
"""
import argparse
import html
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse

from multiview_spec import MAX_VIEWS


_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}')
_ITERATION = re.compile(r'iteration_[0-9]{2,}')
_PAGE = re.compile(r'comparison_page_([0-9]{4,})\.html')
_ORDER = {'computer_lab': 0, 'corridor': 1, 'dorm': 2}
_NAMES = {'computer_lab': '机房（两图共用一个场景）', 'corridor': '走廊', 'dorm': '宿舍'}
_STYLE = '''body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#101318;color:#e7eaf0}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:24px;margin:0}p{max-width:1000px;color:#bec7d5}.lead{font-size:18px}.status,.warning{color:#e9be78}section{padding:26px;background:#1a202a;margin:28px 0;border-radius:12px}article{margin-top:24px}h3{margin-bottom:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}img{width:100%;display:block;border-radius:5px}figcaption{font-size:14px;color:#a5b4c8;margin-bottom:7px}.links,nav{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0}a{color:#80c6ff}details{margin-top:20px}li{margin:8px 0}@media(max-width:760px){main{padding:16px}section{padding:16px}.pair{grid-template-columns:1fr}}'''


def _local(root, *parts):
    """Resolve internal artifacts without following paths outside their owner."""
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes its scene/output directory')
    return path


def _identifier(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError('Invalid ' + label)
    return value


def _link(directory, path):
    if path.is_relative_to(directory):
        value = urllib.parse.quote(path.relative_to(directory).as_posix(), safe='/')
    else:
        value = path.as_uri()
    return html.escape(value, quote=True)


def _source(folder, iteration, value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Invalid source image path')
    path = Path(value)
    # Absolute original-photo paths are an explicit supported dependency.
    # Relative references use the bundle's directory and cannot escape the job.
    return path.resolve() if path.is_absolute() else _local(folder, iteration, path)


def _read_object(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(path.name + ' must contain a JSON object')
    return value


def _output_path(directory, name):
    target = directory / name
    if target.is_symlink():
        raise ValueError('Report output must not be a symlink')
    return _local(directory, name)


def _atomic_write(directory, name, content):
    target = _output_path(directory, name)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.' + name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _contact_row(scene_id, view_id, original, rendered):
    from PIL import Image, ImageDraw, ImageOps
    row = Image.new('RGB', (1600, 640), (27, 30, 34))
    draw = ImageDraw.Draw(row)
    unavailable = []
    for x, path, label in [(0, original, 'REFERENCE'), (800, rendered, 'BATCH RENDER')]:
        try:
            with Image.open(path) as picture:
                im = ImageOps.exif_transpose(picture).convert('RGB')
                im.thumbnail((790, 592))
                row.paste(im, (x + (800 - im.width) // 2, 40 + (592 - im.height) // 2))
                im.close()
        except (OSError, ValueError, Image.DecompressionBombError):
            draw.text((x + 12, 80), 'Preview unavailable; see scene files.', fill='white')
            unavailable.append(label)
        draw.text((x + 12, 12), f'{scene_id} / {view_id} / {label}', fill='white')
    return row, unavailable


def _scene_card(directory, result, rows, max_contact_views):
    if not isinstance(result, dict):
        raise ValueError('Scene result must be a JSON object')
    scene_id = _identifier(result.get('scene_id'), _ID, 'scene_id')
    title = html.escape(_NAMES.get(scene_id, scene_id))
    raw_folder = directory / scene_id
    if raw_folder.is_symlink():
        raise ValueError('Scene directory must not be a symlink')
    folder = _local(directory, scene_id)
    if not result.get('iteration'):
        return (f'<section><h2>{title}</h2><p>任务未完成：'
                + html.escape(str(result.get('error', ''))) + '</p></section>', 0)
    iteration_id = _identifier(result['iteration'], _ITERATION, 'iteration')
    iteration = _local(folder, iteration_id)
    bundle_path = _local(folder, iteration_id, 'scene_bundle.json')
    # Current results already retain the renderer's compact view records.
    # Reports are unverified diagnostics: using these records avoids parsing
    # all geometry just to find photo paths. Older results still use the bundle.
    # Invalid new records must fail visibly rather than silently falling back.
    raw_views = (result['views'] if 'views' in result
                 else _read_object(bundle_path).get('views'))
    if not isinstance(raw_views, list) or not 1 <= len(raw_views) <= MAX_VIEWS:
        raise ValueError(f'Scene views must contain 1..{MAX_VIEWS} view records')
    review_path = _local(folder, iteration_id, 'review.json')
    review = _read_object(review_path) if review_path.exists() else {}
    review_text = html.escape(str(review.get('summary', '未经过自动视觉审查。')))
    issues = review.get('issues', [])
    if not isinstance(issues, list) or any(not isinstance(item, dict) for item in issues):
        raise ValueError('Review issues must be a list of objects')
    # Validate every path before decoding any images for this scene.
    views, seen = [], set()
    for view in raw_views:
        if not isinstance(view, dict):
            raise ValueError('Invalid view record')
        view_id = _identifier(view.get('id'), _ID, 'view_id')
        if view_id in seen:
            raise ValueError('Duplicate view_id')
        seen.add(view_id)
        original = _source(folder, iteration, view.get('source_image'))
        rendered = _local(folder, iteration_id, f'rgb_{view_id}.png')
        annotation = _local(folder, iteration_id, f'annotations_{view_id}.json')
        views.append((view_id, original, rendered, annotation))
    blend = _local(folder, iteration_id, 'scene.blend')
    provenance = _local(folder, 'analysis_provenance.json')
    blocks, view_count = [], 0
    for view_id, original, rendered, annotation in views:
        if not rendered.is_file():
            blocks.append(f'<p class="warning">{html.escape(view_id)}：渲染图缺失。</p>')
            continue
        view_count += 1
        warning = ''
        if len(rows) < max_contact_views:
            row, unavailable = _contact_row(scene_id, view_id, original, rendered)
            rows.append(row)
            if unavailable:
                warning = '<p class="warning">拼图预览不可用：' + ', '.join(unavailable) + '。</p>'
        original_link, rendered_link = _link(directory, original), _link(directory, rendered)
        blocks.append(f'''<article><h3>{html.escape(view_id)}</h3>{warning}<div class="pair">
<figure><figcaption>原照片</figcaption><a href="{original_link}"><img loading="lazy" src="{original_link}"></a></figure>
<figure><figcaption>批量生成 · {iteration_id}</figcaption><a href="{rendered_link}"><img loading="lazy" src="{rendered_link}"></a></figure></div>
<div class="links"><a href="{_link(directory, annotation)}">相机与对象标注</a></div></article>''')
    issue_list = ''.join('<li>' + html.escape(str(item.get('evidence', ''))) + '</li>' for item in issues[:6])
    card = f'''<section><h2>{title}</h2><p class="status">状态：{html.escape(str(result.get('status', 'unknown')))}</p>
<p>{review_text}</p><div class="links"><a href="{_link(directory, blend)}">下载可编辑 Blender 文件</a><a href="{_link(directory, bundle_path)}">最终场景参数</a><a href="{_link(directory, provenance)}">自动分析来源记录</a></div>
{''.join(blocks)}<details><summary>自动审查发现的差异</summary><ul>{issue_list}</ul></details></section>'''
    return card, view_count


def _page_name(number):
    return 'comparison.html' if number == 1 else f'comparison_page_{number:04d}.html'


def _navigation(number, count):
    if count == 1:
        return ''
    links = [f'<span>第 {number} / {count} 页</span>']
    if number > 1:
        links.extend([f'<a href="{_page_name(1)}">首页</a>',
                      f'<a href="{_page_name(number - 1)}">上一页</a>'])
    if number < count:
        links.extend([f'<a href="{_page_name(number + 1)}">下一页</a>',
                      f'<a href="{_page_name(count)}">末页</a>'])
    return '<nav aria-label="场景分页">' + ''.join(links) + '</nav>'


def report(directory, max_contact_views=24, page_size=50):
    from PIL import Image
    for value, name, upper in ((max_contact_views, 'max_contact_views', 100), (page_size, 'page_size', 500)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f'{name} must be 1..{upper}')
    directory = Path(directory).resolve()
    _local(directory, 'summary.json')
    from batch_index import read_snapshot
    summary = read_snapshot(directory)
    scenes = summary.get('scenes')
    if not isinstance(scenes, list):
        raise ValueError('summary.json scenes must be a list')
    def order(record):
        scene_id = record.get('scene_id') if isinstance(record, dict) else None
        return _ORDER.get(scene_id, 10) if isinstance(scene_id, str) else 10
    scenes = sorted(scenes, key=order)
    page_count = max(1, (len(scenes) + page_size - 1) // page_size)
    rows = []
    try:
        for number in range(1, page_count + 1):
            cards, view_count = [], 0
            for result in scenes[(number - 1) * page_size:number * page_size]:
                try:
                    card, rendered_count = _scene_card(directory, result, rows, max_contact_views)
                    view_count += rendered_count
                except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError) as exc:
                    title = str(result.get('scene_id', '无效场景记录')) if isinstance(result, dict) else '无效场景记录'
                    card = (f'<section class="report-error"><h2>{html.escape(title)}</h2>'
                            f'<p>该场景报告不可用：{html.escape(str(exc))}</p></section>')
                cards.append(card)
            navigation = _navigation(number, page_count)
            prefix = '本页 ' if page_count > 1 else ''
            page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>批量照片重建 · 原图对照</title>
<style>{_STYLE}</style><main><h1>批量场景 · 原照片与最终效果</h1>
<p>{len(scenes)} 个场景 · {prefix}{view_count} 个已渲染视角。对比总图最多展示前 {max_contact_views} 个视角，网页按每页 {page_size} 个场景展示全部结果。</p>{navigation}
<p class="lead">左侧为原照片，右侧为最终 Blender 渲染。同组照片共用一套几何、材质和灯光，每张照片保留对应相机。</p>
<p>尺寸、遮挡后结构、相机和光照均为估计。needs_review 表示仍有明显差异；generated_unreviewed 表示未验收。生成成功不等于通过还原验收。点击图片查看原尺寸。</p>
{''.join(cards)}{navigation}<p><a href="summary.json">批量汇总</a> · <a href="groups.json">输入分组</a> · <a href="comparison.jpg">对照拼图（最多 {max_contact_views} 个视角）</a></p></main></html>'''
            _atomic_write(directory, _page_name(number), page)
        # Decode only the views actually retained in the contact sheet.
        contact = _output_path(directory, 'comparison.jpg')
        if rows:
            with Image.new('RGB', (1600, len(rows) * 640), (27, 30, 34)) as sheet:
                for index, row in enumerate(rows):
                    sheet.paste(row, (0, index * 640))
                sheet.save(contact, quality=94)
        else:
            contact.unlink(missing_ok=True)
        for path in directory.iterdir():
            match = _PAGE.fullmatch(path.name)
            if match and int(match[1]) > page_count:
                # Remove only our obsolete page files, never linked targets.
                if path.is_symlink() or path.is_file():
                    path.unlink()
    finally:
        for row in rows:
            row.close()
    return directory / 'comparison.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--max-contact-views', type=int, default=24)
    parser.add_argument('--page-size', type=int, default=50)
    args = parser.parse_args()
    print(report(args.directory, args.max_contact_views, args.page_size))
