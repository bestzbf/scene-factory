"""Offline, resumable intake of the two-file static-asset protocol.

Technical admission deliberately does not assert semantic completeness, visual
acceptance, licensing, or physical stability. Original inputs are never edited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

if __package__:
    from . import _enable_legacy_imports
    _enable_legacy_imports()

INPUT_VERSION = 'scene_asset_input_v1'
RECEIPT_VERSION = 'asset_intake_receipt_v1'
POLICY_VERSION = 'asset_intake_v1'
FIELDS = {'schema_version', 'category', 'placement', 'longest_dimension_m', 'scale_source'}


def _json(path):
    if Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError('JSON asset metadata exceeds 16 MiB')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value): raise ValueError('nonfinite JSON value')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)


def validate_input(data):
    if not isinstance(data, dict) or set(data) != FIELDS: raise ValueError('asset.json requires exactly the five scene_asset_input_v1 fields')
    if data['schema_version'] != INPUT_VERSION: raise ValueError('unsupported asset input version')
    if not isinstance(data['category'], str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}', data['category']): raise ValueError('category must be lower snake_case')
    if data['placement'] not in ('floor', 'surface'): raise ValueError('placement must be floor or surface')
    length = data['longest_dimension_m']
    if type(length) not in (int,float) or not math.isfinite(length) or not 0 < length <= 1000: raise ValueError('longest_dimension_m must be finite and positive, at most 1000')
    if data['scale_source'] not in ('measured','known_reference','estimated'): raise ValueError('invalid scale_source')
    return data


def _safe_tree_path(root, path):
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink(): raise ValueError('asset inputs must not contain symlinks')
    return path


def _record_for(asset_id, metadata, receipt, model_file, receipt_file, receipt_hash):
    asset = dict(receipt['asset'])
    asset.update(model_file=model_file, manifest_file=receipt_file,
        audit_provenance={'intake_gate': POLICY_VERSION, 'manifest_gate': False,
                          'manifest_sha256': receipt_hash, 'quality_status': 'technical_checked',
                          'visual_acceptance': False, 'scale_source': metadata['scale_source']})
    variant = dict(receipt['material'])
    variant['audit_provenance'] = {'intake_gate': POLICY_VERSION, 'manifest_sha256': receipt_hash}
    return asset, variant


def ingest_library(input_dir, output_dir, *, blender='blender', timeout=300, progress=None):
    """Return report; bad assets are quarantined individually and resume is hashed.

    output_dir must be outside input_dir. One Blender subprocess per asset keeps
    failures and datablock state isolated. No network/model calls are made.
    """
    source = Path(input_dir).expanduser().resolve(); output = Path(output_dir).expanduser().resolve()
    if not source.is_dir(): raise ValueError('input asset directory does not exist')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('source and output libraries must be disjoint directory trees')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout <= 0: raise ValueError('invalid Blender timeout')
    output.mkdir(parents=True, exist_ok=True)
    from asset_catalog import AssetCatalog
    AssetCatalog._checked_hash_path(output)
    worker = Path(__file__).with_name('asset_intake_blender.py')
    fingerprint_code = sha(worker) + sha(Path(__file__))
    paths = sorted(source.rglob('asset.json'))
    if not paths: raise ValueError('No asset.json packages found')
    counts = {}
    for path in paths: counts[path.parent.name] = counts.get(path.parent.name,0)+1
    records=[]; assets=[]; materials={}
    import fcntl
    with (output/'.intake.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('another intake is already using this output directory') from None
        for path in paths:
            asset_id=path.parent.name
            record={'asset_id':asset_id,'source':str(path.parent),'status':'quarantined'}
            temporary = None
            try:
                _safe_tree_path(source,path); model=_safe_tree_path(source,path.with_name('model.glb'))
                if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,119}',asset_id): raise ValueError('package directory must be a safe unique asset ID')
                if counts[asset_id] != 1: raise ValueError('duplicate package directory/asset ID')
                if not model.is_file(): raise ValueError('model.glb is missing')
                metadata=validate_input(_json(path))
                hashes={'model.glb':sha(model),'asset.json':sha(path)}
                fingerprint=hashlib.sha256(json.dumps([hashes,fingerprint_code],sort_keys=True).encode()).hexdigest()
                target=output/'assets'/asset_id; receipt_path=target/'receipt.json'
                reused=False
                if receipt_path.is_file():
                    receipt=_json(receipt_path)
                    if receipt.get('input_fingerprint') == fingerprint:
                        for name,digest in receipt.get('output_hashes',{}).items():
                            if name not in ('model.glb','source_asset.json'): raise ValueError('unsafe committed receipt output')
                            if sha(target/name)!=digest: raise ValueError('committed intake output hash changed')
                        if set(receipt.get('output_hashes',{})) != {'model.glb','source_asset.json'}: raise ValueError('incomplete committed intake outputs')
                        reused=True
                    else: raise ValueError('asset ID already exists with changed input or intake implementation; choose a new library version')
                if not reused:
                    temporary=Path(tempfile.mkdtemp(prefix='.intake-',dir=output))
                    shutil.copyfile(path,temporary/'source_asset.json')
                    started=time.monotonic()
                    with (temporary/'blender.log').open('wb') as log:
                        subprocess.run([str(blender),'-b','--factory-startup','--python-exit-code','1','--python',str(worker),'--',
                            '--input',str(model),'--metadata',str(temporary/'source_asset.json'),'--output-dir',str(temporary),
                            '--asset-id',asset_id],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
                    if sha(model)!=hashes['model.glb'] or sha(path)!=hashes['asset.json']: raise ValueError('source asset changed during intake')
                    receipt=_json(temporary/'worker_receipt.json')
                    receipt.update(input_fingerprint=fingerprint,source_hashes=hashes,elapsed_seconds=round(time.monotonic()-started,3),
                        output_hashes={name:sha(temporary/name) for name in ('model.glb','source_asset.json')})
                    atomic_json(temporary/'receipt.json',receipt)
                    (temporary/'worker_receipt.json').unlink()
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists(): raise ValueError('uncommitted asset output already exists; inspect before retrying')
                    os.replace(temporary,target);temporary=None
                model_relative=(target/'model.glb').relative_to(output).as_posix()
                asset,material=_record_for(asset_id,metadata,receipt,model_relative,receipt_path.relative_to(output).as_posix(),sha(receipt_path))
                assets.append(asset);materials[material['material_variant_id']]=material
                record.update(status='technical_ready',reused=reused,visual_acceptance=False,
                              model_file=model_relative,dimensions_m=asset['dimensions_import_units'])
            except Exception as exc:
                record['error']=str(exc)
                quarantine=output/'quarantine';quarantine.mkdir(exist_ok=True)
                if temporary is not None:
                    log=temporary/'blender.log'
                    if log.is_file(): shutil.copyfile(log,quarantine/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.log'))
                atomic_json(quarantine/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.json'),record)
            finally:
                if temporary is not None: shutil.rmtree(temporary,ignore_errors=True)
            records.append(record)
            if progress: progress(record)
            report={'schema_version':'asset_intake_report_v1','source':str(source),'output':str(output),'assets':records,
                    'technical_ready':sum(r['status']=='technical_ready' for r in records),
                    'quarantined':sum(r['status']=='quarantined' for r in records),'visual_accepted':0,'api_requests':0}
            atomic_json(output/'intake_report.json',report)
        catalog={'schema_version':'asset_catalog_v1','root':str(output),'assets':assets,'material_variants':materials,
                 'admission_policy':{'mode':'technical_only','visual_acceptance':False,'intake_version':POLICY_VERSION}}
        # Structural validation plus the receipt gate catches incompatible
        # compiler output before publishing a catalog used by scene workers.
        candidate=AssetCatalog(catalog)
        for asset in assets:
            ready,reasons=candidate.production_readiness(asset)
            if not ready: raise ValueError('compiled asset failed admission: '+asset['asset_id']+': '+', '.join(reasons))
        atomic_json(output/'asset_catalog.json',catalog)
        report['catalog']=str(output/'asset_catalog.json');atomic_json(output/'intake_report.json',report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--blender',default='blender');parser.add_argument('--timeout',type=float,default=300)
    args=parser.parse_args(argv)
    print(json.dumps(ingest_library(args.input_dir,args.output_dir,blender=args.blender,timeout=args.timeout),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
