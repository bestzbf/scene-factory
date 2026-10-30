"""Repeat/spacing semantics shared by validation, support routing and Blender expansion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch import file_sha256, read_json
from schema import expand_repeat_positions, validate as validate_v1
from spec_v2 import material, part, upgrade_v1, validate as validate_v2
import vision_v2


ROOT = Path(__file__).parent


def base_spec():
    return json.loads((ROOT / 'examples/living_room.json').read_text())


class RepeatSpacingTests(unittest.TestCase):
    def test_active_zero_spacing_is_rejected_on_both_axes(self):
        for repeat, axis in (([2, 1], 'x'), ([1, 2], 'y')):
            scene = base_spec()
            scene['objects'][0]['repeat'] = repeat
            scene['objects'][0]['spacing'] = [0, 0]
            with self.subTest(repeat=repeat), self.assertRaisesRegex(
                    ValueError, rf'scene\.objects\[0\]\.spacing\.{axis}.*repeat\.{axis} > 1'):
                validate_v1(scene)

            v2 = upgrade_v1(base_spec())
            v2['base_scene']['objects'][0]['repeat'] = repeat
            v2['base_scene']['objects'][0]['spacing'] = [0, 0]
            with self.subTest(v2_repeat=repeat), self.assertRaisesRegex(
                    ValueError, rf'scene\.objects\[0\]\.spacing\.{axis}.*repeat\.{axis} > 1'):
                validate_v2(v2)

    def test_zero_spacing_is_allowed_on_inactive_axes(self):
        scene = base_spec()
        scene['objects'][0]['repeat'] = [2, 1]
        scene['objects'][0]['spacing'] = [1.25, 0]
        self.assertIs(validate_v1(scene), scene)

        v2 = upgrade_v1(scene)
        self.assertIs(validate_v2(v2), v2)
        self.assertEqual(expand_repeat_positions([0, 0, 0], [2, 1], [1.25, 0]), [
            (0, 0, [0, 0, 0]), (1, 0, [1.25, 0, 0])])

    def test_negative_spacing_is_valid_and_steps_in_negative_world_direction(self):
        scene = base_spec()
        entry = scene['objects'][0]
        entry['position'] = [2, 3, .5]
        entry['repeat'] = [2, 2]
        entry['spacing'] = [-1.5, -2]
        self.assertIs(validate_v1(scene), scene)
        self.assertEqual(expand_repeat_positions(entry['position'], entry['repeat'], entry['spacing']), [
            (0, 0, [2, 3, .5]),
            (0, 1, [2, 1, .5]),
            (1, 0, [.5, 3, .5]),
            (1, 1, [.5, 1, .5]),
        ])

    def test_expanded_positions_are_unique_and_repeat_axes_are_world_xy(self):
        instances = expand_repeat_positions([1, 2, .75], [2, 2], [-1, 3])
        positions = [tuple(position) for _, _, position in instances]
        self.assertEqual(len(positions), 4)
        self.assertEqual(len(set(positions)), len(positions))
        self.assertEqual(instances, [
            (0, 0, [1, 2, .75]),
            (0, 1, [1, 5, .75]),
            (1, 0, [0, 2, .75]),
            (1, 1, [0, 5, .75]),
        ])

    def test_glm_desk_mid1_riser_fails_precisely_before_ambiguous_support_resolution(self):
        scene = upgrade_v1(base_spec())
        desk = copy.deepcopy(scene['base_scene']['objects'][0])
        desk.update(id='desk_mid1', position=[-.4, 3.4, .75], repeat=[2, 1], spacing=[0, 2])
        scene['base_scene']['objects'] = [desk]
        scene['materials'] = [material('desk_surface', [.5, .4, .3])]
        scene['custom_objects'] = [{
            'id': 'riser_mid_front', 'position': [.4, 2.4, 1.12],
            'rotation_degrees': [0, 0, 190], 'material_id': 'desk_surface',
            'confidence': .8, 'observed': True, 'support_id': 'desk_mid1',
            'parts': [part('box', position=[0, 0, .35], size=[1.8, .35, .04])],
        }]
        with self.assertRaisesRegex(
                ValueError,
                r'base_scene\.objects\[0\]\.spacing\.x.*repeat\.x > 1'):
            vision_v2.resolve_support_groups(scene)

    def test_single_view_analysis_records_material_expansion_separately_from_supports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.jpg'
            source.write_bytes(b'local test image bytes')
            scene = upgrade_v1(base_spec(), width=40, height=20,
                               source_id=file_sha256(source)[:16])
            desk = scene['base_scene']['objects'][0]
            desk.update(id='repeat_desk', repeat=[2, 1], spacing=[1.2, 0])
            scene['materials'] = [material('wood_override', [.4, .25, .12])]
            scene['base_material_overrides'] = [{'object_id': 'repeat_desk',
                'material_id': 'wood_override',
                'selector': {'role': 'body', 'material_name': '', 'slot_index': -1}}]
            output = root / 'scene_spec.json'
            with patch.object(vision_v2, 'call_model', return_value=scene):
                result = vision_v2.analyze(source, output, width=40, height=20)
            provenance = read_json(root / 'analysis_provenance.json')['normalizations']
            self.assertEqual(provenance['supports'], [])
            self.assertEqual(provenance['base_material_overrides'][0]['from'], 'repeat_desk')
            self.assertEqual([item['object_id'] for item in result['base_material_overrides']],
                             ['repeat_desk__0_0', 'repeat_desk__1_0'])


if __name__ == '__main__':
    unittest.main()
