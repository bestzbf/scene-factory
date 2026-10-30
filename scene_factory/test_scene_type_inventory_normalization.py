"""High-confidence local scene-type correction from expanded inventory."""

import copy
import json
import unittest
from pathlib import Path

from spec_v2 import upgrade_v1, validate as validate_v2
from scene_type_normalization import normalize_scene_type_inventory
from test_multiview import fixture as multiview_fixture
import vision_multiview


def office_like_shop():
    base = json.loads((Path(__file__).parent / 'examples' / 'living_room.json').read_text())
    spec = upgrade_v1(base, width=640, height=480)
    spec['scene_type'] = 'shop'
    spec['base_scene']['scene_type'] = 'other'
    template = {
        'position': [0, 2, 0], 'dimensions': [1, 1, .75], 'rotation_z_deg': 0,
        'color': [.5, .5, .5], 'repeat': [1, 1], 'spacing': [0, 0], 'confidence': .9,
    }
    spec['base_scene']['objects'] = [
        {**template, 'id': 'desk_row', 'category': 'desk', 'repeat': [2, 1], 'spacing': [1.5, 0]},
        {**template, 'id': 'screens', 'category': 'monitor', 'repeat': [2, 1], 'spacing': [1.5, 0]},
        {**template, 'id': 'seats', 'category': 'chair', 'repeat': [2, 1], 'spacing': [1.5, 0]},
    ]
    return spec


class SceneTypeInventoryTests(unittest.TestCase):
    def test_repeated_inventory_promotes_shop_and_preserves_input(self):
        original = office_like_shop()
        before = copy.deepcopy(original)
        normalized, evidence = normalize_scene_type_inventory(original)
        self.assertEqual(original, before)
        self.assertEqual(normalized['scene_type'], 'office')
        self.assertEqual(normalized['base_scene']['scene_type'], 'office')
        self.assertEqual(evidence['from'], 'shop')
        self.assertEqual(evidence['to'], 'office')
        self.assertEqual(evidence['counts'], {
            'desk': 2, 'monitor': 2, 'computer': 0,
            'monitor_or_computer': 2, 'chair': 2,
        })
        validate_v2(normalized)

    def test_weak_shop_inventory_is_retained(self):
        spec = office_like_shop()
        spec['base_scene']['objects'][0]['repeat'] = [1, 1]
        spec['base_scene']['objects'][1]['repeat'] = [1, 1]
        spec['base_scene']['objects'][1]['spacing'] = [0, 0]
        normalized, evidence = normalize_scene_type_inventory(spec)
        self.assertEqual(normalized['scene_type'], 'shop')
        self.assertEqual(normalized['base_scene']['scene_type'], 'other')
        self.assertEqual(evidence['from'], evidence['to'])

    def test_multiview_normalization_applies_shared_rule_and_reports_evidence(self):
        bundle = multiview_fixture()
        shop = office_like_shop()
        bundle['shared_scene'] = shop
        bundle['shared_scene']['base_scene']['camera'] = {
            key: bundle['views'][0]['camera'][key] for key in ('position', 'target', 'lens_mm')}
        bundle['shared_scene']['camera']['roll_degrees'] = bundle['views'][0]['camera']['roll_degrees']
        bundle['shared_scene']['image'] = {'width': 1280, 'height': 960, 'source_id': 'a'}
        raw = copy.deepcopy(bundle)
        for view in raw['views']:
            view['source_image'] = 'input:' + view['id']
        views = [{'id': item['id'], 'source_image': '/trusted/' + item['id'] + '.png',
                  'width': item['width'], 'height': item['height']} for item in raw['views']]
        normalized, provenance = vision_multiview.normalize_response(raw, views, 'room')
        self.assertEqual(normalized['shared_scene']['scene_type'], 'office')
        self.assertEqual(provenance['scene_type']['from'], 'shop')
        self.assertEqual(provenance['scene_type']['to'], 'office')


if __name__ == '__main__':
    unittest.main()
