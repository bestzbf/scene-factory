"""Freeze and run the same local-Codex retail pipeline for any photo groups.

Only the photo manifest, catalog and shared production configuration are inputs.
No store-specific scene/profile/Blender override is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import platform
import importlib.metadata

REPO = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO/'configs/retail/codex_production_v1.json')
    p.add_argument('--effort',choices=['low','medium','high'],default='high')
    p.add_argument('--reuse-analysis-from',type=Path,
                   help='Reuse only verified original Codex responses from this batch; rebuild and review every scene')
    p.add_argument('--prepare-only',action='store_true',
                   help='Freeze all inputs and write command.json; start that exact command when resources are available')
    a=p.parse_args();out=a.out.resolve();a.groups=a.groups.resolve();a.catalog=a.catalog.resolve()
    if out.exists():raise ValueError('Use a new output folder; resume with the saved command.json instead')
    sys.path.insert(0,str(REPO/'scene_factory'))
    from factory_config import load_config
    from production_retail import production_inputs
    from retail_refinement import read_config
    from batch_multiview import load_groups
    load_groups(a.groups)
    c=load_config(a.config)
    if not c.get('local_codex'):raise ValueError('This runner uses the explicitly configured local Codex')
    profile=(a.config.resolve().parent/c['pipeline']['retail_optimization']).resolve()
    production_inputs(profile)
    retail=read_config(profile)
    out.mkdir(parents=True)
    runtime=out/'runtime'
    shutil.copytree(REPO/'scene_factory',runtime/'scene_factory',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    config_dir=runtime/'configs';config_dir.mkdir()
    shutil.copy2(retail['atlas'],config_dir/'packaging.png')
    retail.pop('_config_path',None);retail['atlas']='packaging.png'
    (config_dir/'retail.json').write_text(json.dumps(retail,ensure_ascii=False,indent=2))
    c['pipeline']['retail_optimization']='retail.json'
    c['asset_library']['catalog']=str(a.catalog)
    (config_dir/'factory.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
    shutil.copy2(a.groups,out/'groups.json')
    command=[sys.executable,str(runtime/'scene_factory/batch_multiview.py'),
             '--config',str(config_dir/'factory.json'),'--groups',str(out/'groups.json'),
             '--output-dir',str(out/'batch'),'--effort',a.effort]
    if a.reuse_analysis_from:
        command += ['--reuse-analysis-from', str(a.reuse_analysis_from.resolve())]
    (out/'command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2))
    paths=[p for p in runtime.rglob('*') if p.is_file()]+[out/'groups.json']
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'runtime_manifest.json').write_text(json.dumps(hashes,indent=2))
    software={'python':sys.version,'platform':platform.platform(),
              'packages':{name:importlib.metadata.version(name) for name in ('Pillow','numpy','scipy')}}
    (out/'software_manifest.json').write_text(json.dumps(software,indent=2))
    if a.prepare_only:
        print(json.dumps({'status':'prepared','directory':str(out),'command':str(out/'command.json')},ensure_ascii=False))
        return 0
    return subprocess.call(command)


if __name__=='__main__':raise SystemExit(main())
