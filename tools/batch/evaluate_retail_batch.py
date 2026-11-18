"""Compare all saved-scene batch views against original photographs and baseline.

Scores describe image appearance/structure, not measured 3D reconstruction accuracy.
"""
import argparse
import json
from pathlib import Path
import statistics
from score_alignment import grey,structural


def evaluate(source_root,out):
    records=[];summary={}
    for folder in sorted(out.iterdir()):
        if not folder.is_dir() or not (folder/'verification.json').exists():continue
        bundle=json.loads((source_root/folder.name/'scene_bundle.json').read_text())
        report=json.loads((folder/'retail_refinement.json').read_text())
        rows=[]
        for view in bundle['views']:
            vid=view['id'];new=folder/f'rgb_{vid}.png';old=source_root/folder.name/f'rgb_{vid}.png'
            if not new.exists() or not old.exists():continue
            reference=grey(view['source_image']);before=structural(grey(old),reference);after=structural(grey(new),reference)
            row={'scene_id':folder.name,'view':vid,'before':before,'after':after,'delta':after-before}
            rows.append(row);records.append(row)
        if rows:
            summary[folder.name]={'views':len(rows),'structural_before':statistics.mean(r['before'] for r in rows),
                'structural_after':statistics.mean(r['after'] for r in rows),'improved_views':sum(r['delta']>0 for r in rows),
                'stock_units':report['stock_units'],'added_top_stock':sum(r.get('added_top_stock',0) for r in report['fixtures']),
                'moved_groups':len(report['moves']),'verification':json.loads((folder/'verification.json').read_text())}
    result={'scope':'Gradient correlation/direction against photographs; not a geometry-accuracy metric. Declines are retained.',
            'scenes':summary,'views':records}
    (out/'comparison_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for sid,s in summary.items():print(sid,'views',s['views'],'structure',round(s['structural_before'],4),'->',round(s['structural_after'],4),'improved',s['improved_views'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();evaluate(a.source_root,a.out)
