"""Review a new retail batch against the immediately preceding iteration."""
import argparse
import json
import statistics
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from evaluate_retail_batch import evaluate
from tune_look import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--current',type=Path,required=True)
    a=p.parse_args();result=evaluate(a.previous,a.current);overview=[]
    for sid,scene in result['scenes'].items():
        bundle=json.loads((a.previous/sid/'scene_bundle.json').read_text());rows=[];before=[];after=[]
        for v in bundle['views']:
            vid=v['id'];old=a.previous/sid/f'rgb_{vid}.png';new=a.current/sid/f'rgb_{vid}.png';ref=Path(v['source_image'])
            if not new.exists():continue
            before.append(metrics(old,ref));after.append(metrics(new,ref))
            row=Image.new('RGB',(1440,300),'#202329');d=ImageDraw.Draw(row)
            for col,(label,path) in enumerate([('Reference',ref),('Previous iteration',old),('Current editable 3D',new)]):
                d.text((col*480+8,7),f'{sid} / {vid} / {label}',fill='white')
                with Image.open(path) as im:row.paste(ImageOps.contain(im.convert('RGB'),(474,267)),(col*480+3,28))
            rows.append(row)
        if not rows:continue
        scene['appearance']={k:{'before':statistics.mean(r[k] for r in before),'after':statistics.mean(r[k] for r in after)} for k in ('lum_err','sat_err','ssim')}
        sheet=Image.new('RGB',(1440,300*len(rows)),'#202329')
        for i,row in enumerate(rows):sheet.paste(row,(0,i*300))
        sheet.save(a.current/sid/'previous_vs_current.jpg',quality=92)
        overview.extend([rows[0],rows[-1]])
    sheet=Image.new('RGB',(1440,300*len(overview)),'#202329')
    for i,row in enumerate(overview):sheet.paste(row,(0,i*300))
    sheet.save(a.current/'previous_vs_current.jpg',quality=92)
    result['baseline']=str(a.previous.resolve())
    (a.current/'iteration_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
