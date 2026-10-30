"""Offline policy and hash-gate regression for minimal asset intake."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from asset_intake import validate_input, ingest_library, atomic_json, sha
from asset_catalog import AssetCatalog, CatalogError

class AssetIntakeTests(unittest.TestCase):
    def metadata(self):return {'schema_version':'scene_asset_input_v1','category':'chair','placement':'floor','longest_dimension_m':1.0,'scale_source':'estimated'}
    def test_protocol_rejects_unknown_missing_nonfinite_and_false_scale(self):
        valid=self.metadata();self.assertEqual(validate_input(valid),valid)
        for edit in ({'longest_dimension_m':True},{'longest_dimension_m':float('nan')},{'placement':'wall'}, {'scale_source':'unknown'},{'category':'Chair'}, {'extra':1}):
            with self.subTest(edit=edit),self.assertRaises(ValueError):validate_input({**valid,**edit})
    def compiler(self,command,**kwargs):
        output=Path(command[command.index('--output-dir')+1]);identifier=command[command.index('--asset-id')+1]
        if identifier=='bad':raise RuntimeError('fixture compiler failure')
        (output/'model.glb').write_bytes(b'normalized test model')
        meta=json.loads((output/'source_asset.json').read_text());digest=sha(output/'model.glb')
        asset={'asset_id':identifier,'family_id':'chair','geometry_variant_id':identifier+'__geometry',
               'hash':digest,'status':'technical_ready','dimensions_import_units':[1,.5,.5],
               'meters_per_unit':1,'front_axis':'-Y','up_axis':'+Z','category':'chair','placement':'floor',
               'scope':'single_prop','scale_source':'estimated','material_variant_ids':[identifier+'__native']}
        material={'material_variant_id':identifier+'__native','compatible_geometry_ids':[identifier+'__geometry'],
                  'mode':'embedded_gltf','model_sha256':digest,'maps':{},'slot_materials':{}}
        atomic_json(output/'worker_receipt.json',{'schema_version':'asset_intake_receipt_v1','policy_version':'asset_intake_v1',
            'status':'technical_ready','asset':asset,'material':material,'metadata':meta,'visual_accepted':False,
            'checks':dict.fromkeys(('glb_embedded_standard_opaque','mesh_uv_materials','image_decode','scale_roundtrip'),True)})
    def packages(self,root):
        source=root/'source';source.mkdir()
        for name in ('good','bad'):
            folder=source/name;folder.mkdir();atomic_json(folder/'asset.json',self.metadata());(folder/'model.glb').write_bytes(b'source')
        return source
    def test_batch_quarantines_bad_asset_reuses_valid_and_pins_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.packages(root);output=root/'library'
            with patch('asset_intake.subprocess.run',side_effect=self.compiler) as worker:
                report=ingest_library(source,output)
                self.assertEqual(report['technical_ready'],1);self.assertEqual(report['quarantined'],1);self.assertEqual(worker.call_count,2)
            with patch('asset_intake.subprocess.run',side_effect=self.compiler) as worker:
                resumed=ingest_library(source,output);self.assertEqual(worker.call_count,1)
                self.assertTrue(next(a for a in resumed['assets'] if a['asset_id']=='good')['reused'])
            catalog=AssetCatalog(json.loads((output/'asset_catalog.json').read_text()))
            asset=catalog.get_asset('good');self.assertEqual(catalog.production_readiness(asset),(True,[]))
            changed=copy.deepcopy(asset);changed['category']='table';self.assertFalse(catalog.production_readiness(changed)[0])
            changed=copy.deepcopy(asset);changed['support_surfaces']=[{'z':2}];self.assertFalse(catalog.production_readiness(changed)[0])
            catalog.materials['good__native']['model_sha256']='0'*64
            with self.assertRaises(CatalogError):catalog.verify_material_hashes('good__native')
            catalog=AssetCatalog(json.loads((output/'asset_catalog.json').read_text()))
            (output/'assets/good/model.glb').write_bytes(b'changed model')
            self.assertFalse(catalog.production_readiness('good')[0])
    def test_changed_input_never_overwrites_committed_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.packages(root);output=root/'library'
            with patch('asset_intake.subprocess.run',side_effect=self.compiler):ingest_library(source,output)
            before=(output/'assets/good/model.glb').read_bytes();(source/'good/model.glb').write_bytes(b'changed')
            with patch('asset_intake.subprocess.run',side_effect=self.compiler):report=ingest_library(source,output)
            self.assertEqual(report['technical_ready'],0);self.assertEqual((output/'assets/good/model.glb').read_bytes(),before)
if __name__=='__main__':unittest.main()
