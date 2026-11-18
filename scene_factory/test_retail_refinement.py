"""Pure-Python contract tests; rendered shared-geometry checks live in verification.json."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from retail_refinement import validate_config, support_groups, settings_for, stable_seed, read_config, _move_audit, is_stock_fixture


class RetailContractTests(unittest.TestCase):
    def test_semantic_category_prevents_stock_on_neighbour_named_columns_and_boxes(self):
        self.assertFalse(is_stock_fixture('left_refrigerator_end_column','structural_column'))
        self.assertFalse(is_stock_fixture('carton_by_shelf','cardboard_box'))
        self.assertTrue(is_stock_fixture('left_storage','shelf'))
        self.assertTrue(is_stock_fixture('legacy_shelf_02'))

    def setUp(self):
        self.config={'schema_version':1,'defaults':{'fixtures':{'s':{'kind':'snack'}}},'scenes':{}}

    def test_support_group_transitive_and_cycle(self):
        objects=[{'id':'s'},{'id':'tray','support_id':'s'},{'id':'can','support_id':'tray'},{'id':'other'}]
        self.assertEqual(support_groups(objects)['s'],['s','tray','can'])
        objects[0]['support_id']='can'
        with self.assertRaisesRegex(ValueError,'Cyclic'):support_groups(objects)
        with self.assertRaisesRegex(ValueError,'Duplicate'):support_groups([{'id':'s'},{'id':'s'}])

    def test_seed_and_overrides_do_not_mutate_defaults(self):
        self.config['scenes']['a']={'fixtures':{'s':{'kind':'beverage'}}}
        original=copy.deepcopy(self.config)
        self.assertEqual(settings_for(self.config,'a')['fixtures']['s']['kind'],'beverage')
        self.assertEqual(settings_for(self.config,'b')['fixtures']['s']['kind'],'snack')
        self.assertEqual(self.config,original)
        self.assertEqual(stable_seed('a','s'),stable_seed('a','s'))
        self.assertNotEqual(stable_seed('a','s'),stable_seed('b','s'))

    def test_invalid_moves_and_kind(self):
        for v in ([float('nan'),0,0],[3,0,0],[0,0],['bad',0,0]):
            c=copy.deepcopy(self.config);c['defaults']['moves']=[{'id':'s','translation_m':v}]
            with self.assertRaises(ValueError):validate_config(c)
        self.config['defaults']['fixtures']['s']['kind']='unknown'
        with self.assertRaises(ValueError):validate_config(self.config)

    def test_relative_atlas_missing_fails_early(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.json';self.config['atlas']='missing.png';p.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError,'atlas is missing'):read_config(p)
            (Path(t)/'missing.png').write_bytes(b'placeholder')
            self.assertEqual(read_config(p)['atlas'],str((Path(t)/'missing.png').resolve()))

    def test_visual_config_rejects_invalid_physical_dimensions(self):
        for field, value in [('floor_tiles', {'color':[.2,.2,.2], 'size_m':0}),
                             ('entrance', {'width_m':float('inf')}),
                             ('floor_tiles', {'color':[1.2,.2,.2]})]:
            c=copy.deepcopy(self.config);c['defaults'][field]=value
            with self.assertRaises(ValueError):validate_config(c)
        c=copy.deepcopy(self.config)
        c['defaults']['fixtures']['s']['shelf_edges']={'color':[.1,.4,.1],'height_m':.028}
        c['defaults']['floor_tiles']={'color':[.2,.2,.2],'size_m':.6}
        self.assertIs(validate_config(c),c)

    def test_clearance_rejects_new_overlap_but_allows_old(self):
        room={'width':6,'depth':10,'height':3}
        before={'a':([-2,1,0],[-1,2,2]),'b':([0,1,0],[1,2,2])}
        self.assertEqual(_move_audit(before,before,{'a'},room)['status'],'passed')
        after={**before,'a':([-.5,1,0],[.5,2,2])}
        self.assertEqual(_move_audit(before,after,{'a'},room)['status'],'rejected')
        after={**before,'a':([-4,1,0],[-3,2,2])}
        self.assertEqual(_move_audit(before,after,{'a'},room)['status'],'rejected')


if __name__=='__main__':unittest.main()
