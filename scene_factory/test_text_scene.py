"""Text-authored scenes: schema, plausibility rules and layout expansion.

These tests are deliberately model-free. The whole point of the text mode is
that its correctness is checkable without a provider: a layout either expands
into a valid V3 bundle or it does not, and a scene either breaks a physical
rule or it does not.
"""
import copy
import json
import unittest

import scene_logic
import text_scene
from scene_logic import validate_scene_logic
from text_scene import (FORMAT_VERSION, expand_to_bundle, schema_for,
                        validate_text_scene)


def room(width=5.0, depth=4.2, height=2.7):
    return {'width': width, 'depth': depth, 'height': height,
            'wall_color': [.86, .85, .82], 'floor_color': [.55, .42, .30],
            'ceiling_color': [.95, .95, .94], 'style': 'warm minimal'}


def entry(identifier, category, xy, dims, yaw=0, support='floor', clearance=0.0,
          supports=True, note=''):
    return {'id': identifier, 'category': category, 'xy': list(xy),
            'target_dimensions_m': list(dims), 'rotation_z_deg': yaw,
            'support_id': support, 'clearance_m': clearance,
            'supports_objects': supports, 'note': note}


def living_room():
    """A scene that should pass: sofa facing the television, rug underneath."""
    return {
        'schema_version': FORMAT_VERSION,
        'scene_id': 'demo-living',
        'scene_type': 'living_room',
        'description': 'A calm living room: a grey sofa faces a low television unit, '
                       'with a coffee table between them and a rug beneath.',
        'seed': 7,
        'room': room(),
        'lighting': {'energy': 900, 'color': [1, .96, .9], 'world_strength': .35},
        'lights': [],
        'exposure': 0.0,
        'world_color': [.8, .85, .9],
        'cameras': [
            {'position': [0, .45, 1.55], 'target': [0, 2.8, 1.0], 'lens_mm': 24, 'roll_degrees': 0},
            {'position': [-1.9, 3.7, 1.6], 'target': [.4, 1.0, .9], 'lens_mm': 28, 'roll_degrees': 0},
        ],
        'objects': [
            entry('sofa_01', 'sofa', (0, .55), (2.1, .9, .8), yaw=180),
            entry('coffee_table_01', 'coffee_table', (0, 1.9), (1.1, .6, .4)),
            entry('tv_unit_01', 'tv', (0, 3.85), (1.6, .4, .5), yaw=180),
            entry('rug_01', 'rug', (0, 1.5), (2.4, 1.8, .02), supports=False),
            entry('floor_lamp_01', 'floor_lamp', (-2.1, .5), (.4, .4, 1.65), supports=False),
        ],
        'assumptions': ['no window modelled'],
    }


class SchemaTests(unittest.TestCase):
    def test_schema_binds_identity_and_camera_count(self):
        rule = schema_for('my-scene', 3)
        self.assertEqual(rule['properties']['scene_id']['enum'], ['my-scene'])
        cameras = rule['properties']['cameras']
        self.assertEqual((cameras['minItems'], cameras['maxItems']), (3, 3))

    def test_schema_rejects_bad_scene_id(self):
        with self.assertRaises(ValueError):
            schema_for('bad id with spaces', 2)
        with self.assertRaises(ValueError):
            schema_for('ok', 0)

    def test_scene_type_is_required_by_schema(self):
        # This is the contract the photo pipeline forgot: the enum is required,
        # so the prompt has to mention it or the model cannot satisfy it.
        self.assertIn('scene_type', text_scene.SCHEMA['required'])
        self.assertIn('scene_type', text_scene.CONTRACT)
        for name in text_scene.SCHEMA['properties']['scene_type']['enum']:
            self.assertIn(name, text_scene.CONTRACT)

    def test_validate_accepts_a_coherent_scene(self):
        validate_text_scene(living_room())

    def test_validate_rejects_object_outside_room(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [40, .5]
        with self.assertRaises(ValueError):
            validate_text_scene(scene)

    def test_validate_rejects_unknown_support(self):
        scene = living_room()
        scene['objects'][1]['support_id'] = 'ghost_table'
        with self.assertRaises(ValueError):
            validate_text_scene(scene)

    def test_validate_rejects_cyclic_supports(self):
        scene = living_room()
        scene['objects'] = [
            entry('a', 'table', (0, 1), (1, 1, .4), support='b'),
            entry('b', 'table', (0, 2), (1, 1, .4), support='a'),
        ]
        with self.assertRaises(ValueError):
            validate_text_scene(scene)

    def test_validate_rejects_floor_clearance(self):
        scene = living_room()
        scene['objects'][0]['clearance_m'] = .03
        with self.assertRaises(ValueError):
            validate_text_scene(scene)

    def test_validate_rejects_camera_outside_room(self):
        scene = living_room()
        scene['cameras'][0]['position'] = [0, 0, 9.0]
        with self.assertRaises(ValueError):
            validate_text_scene(scene)


class ExpansionTests(unittest.TestCase):
    def test_expansion_keeps_identity_and_camera_binding(self):
        bundle, _ = expand_to_bundle(living_room())
        self.assertEqual(bundle['schema_version'], 3)
        self.assertEqual(len(bundle['views']), 2)
        self.assertEqual(bundle['views'][0]['camera']['position'],
                         bundle['shared_scene']['base_scene']['camera']['position'])
        self.assertEqual(bundle['views'][0]['camera']['roll_degrees'],
                         bundle['shared_scene']['camera']['roll_degrees'])

    def test_expansion_resolves_support_height(self):
        scene = living_room()
        scene['objects'].append(
            entry('monitor_01', 'monitor', (0, 1.9), (.5, .2, .4), support='coffee_table_01'))
        bundle, _ = expand_to_bundle(scene)
        objects = {o['id']: o for o in bundle['shared_scene']['base_scene']['objects']}
        # The monitor sits on a 0.40 m table, so its bottom is at 0.40 m.
        self.assertAlmostEqual(objects['monitor_01']['position'][2], .40, places=6)
        self.assertAlmostEqual(objects['coffee_table_01']['position'][2], 0.0, places=6)

    def test_expansion_requires_source_reference_paths(self):
        bundle, _ = expand_to_bundle(living_room())
        for view in bundle['views']:
            self.assertTrue(view['source_image'].startswith('references/'))
            self.assertEqual(view['width'], 1600)

    def test_expansion_does_not_mutate_input(self):
        scene = living_room()
        before = json.dumps(scene, sort_keys=True)
        expand_to_bundle(scene)
        self.assertEqual(json.dumps(scene, sort_keys=True), before)


class LogicTests(unittest.TestCase):
    def codes(self, scene):
        return {f['code'] for f in validate_scene_logic(scene)['findings']}

    def test_coherent_scene_is_plausible(self):
        report = validate_scene_logic(living_room())
        self.assertTrue(report['plausible'], report['findings'])
        self.assertEqual(report['blocking'], 0)

    def test_rug_under_sofa_is_not_a_collision(self):
        # Regression: the first version of this check reported every rug as
        # interpenetrating the furniture standing on it.
        self.assertNotIn('object.interpenetrates', self.codes(living_room()))

    def test_sofa_facing_away_from_television_is_flagged(self):
        scene = living_room()
        scene['objects'][0]['rotation_z_deg'] = 0        # now faces -Y, away from the TV
        self.assertIn('seating.faces_nothing', self.codes(scene))

    def test_implausible_height_is_flagged(self):
        scene = living_room()
        scene['objects'][1]['target_dimensions_m'] = [1.1, .6, 3.0]
        self.assertIn('object.height_implausible', self.codes(scene))

    def test_overlapping_furniture_is_blocking(self):
        scene = living_room()
        scene['objects'][1]['xy'] = [0, .55]              # inside the sofa
        report = validate_scene_logic(scene)
        self.assertIn('object.interpenetrates', {f['code'] for f in report['findings']})
        self.assertGreaterEqual(report['blocking'], 1)

    def test_object_outside_room_is_blocking(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [0, .05]              # sofa half through the front wall
        report = validate_scene_logic(scene)
        self.assertIn('object.outside_room', {f['code'] for f in report['findings']})
        self.assertGreater(report['blocking'], 0)

    def test_coincident_objects_flagged_but_support_pairs_exempt(self):
        scene = living_room()
        scene['objects'].append(
            entry('lamp_01', 'desk_lamp', (0, 1.9), (.3, .3, .5), support='coffee_table_01'))
        # Sharing a position with its own support is legitimate.
        self.assertNotIn('object.coincident', self.codes(scene))
        # Two floor objects at one point is not.
        scene['objects'][4]['xy'] = [.4, 1.9]
        scene['objects'].append(entry('extra_01', 'chair', (.4, 1.9), (.5, .5, .9)))
        self.assertIn('object.coincident', self.codes(scene))

    def test_doorway_blocked_is_flagged(self):
        scene = living_room()
        scene['objects'].append(entry('door_01', 'door', (0, .05), (.9, .1, 2.1), supports=False))
        self.assertIn('circulation.blocks_doorway', self.codes(scene))

    def test_child_wider_than_parent_is_flagged(self):
        scene = living_room()
        scene['objects'].append(
            entry('big_01', 'cabinet', (0, 1.9), (2.0, 1.0, .8), support='coffee_table_01'))
        self.assertIn('support.child_larger_than_parent', self.codes(scene))

    def test_overcrowded_room_is_flagged(self):
        scene = living_room()
        scene['room'] = room(3.0, 3.0)
        for index in range(14):
            scene['objects'].append(
                entry(f'box_{index}', 'box', (-1 + index % 4, .4 + index // 4),
                      (.4, .4, .4), supports=False))
        self.assertIn('room.overcrowded', self.codes(scene))

    def test_report_is_json_serialisable(self):
        report = validate_scene_logic(living_room())
        json.dumps(report)
        self.assertEqual(report['schema_version'], 'scene_logic_report_v1')
        self.assertIn('scope', report)

    def test_strict_promotes_geometry_findings_to_blocking(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [0, .05]
        loose = validate_scene_logic(scene, strict=False)
        strict = validate_scene_logic(scene, strict=True)
        self.assertEqual(len(loose['findings']), len(strict['findings']))
        self.assertGreaterEqual(strict['blocking'], loose['blocking'])

    def test_repair_hint_names_only_the_worst_problems(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [0, .05]
        scene['objects'][1]['target_dimensions_m'] = [1.1, .6, 3.0]
        hint = scene_logic.repair_hint(validate_scene_logic(scene))
        self.assertIsNotNone(hint)
        self.assertIn('sofa_01', hint)

    def test_repair_hint_is_none_for_a_clean_scene(self):
        self.assertIsNone(scene_logic.repair_hint(validate_scene_logic(living_room())))


class SafeFixTests(unittest.TestCase):
    def test_out_of_room_object_is_pulled_back(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [0, -.4]
        fixed, changes = scene_logic.apply_safe_fixes(scene)
        self.assertTrue(any(c['action'] == 'moved_inside_room' for c in changes))
        self.assertGreaterEqual(fixed['objects'][0]['xy'][1], 0)

    def test_safe_fixes_leave_a_valid_scene_untouched(self):
        scene = living_room()
        fixed, changes = scene_logic.apply_safe_fixes(scene)
        self.assertEqual(changes, [])
        self.assertEqual(fixed['objects'], scene['objects'])

    def test_safe_fixes_do_not_mutate_the_input(self):
        scene = living_room()
        scene['objects'][0]['xy'] = [0, -.4]
        before = copy.deepcopy(scene)
        scene_logic.apply_safe_fixes(scene)
        self.assertEqual(scene, before)


class PhotoContractRegressionTests(unittest.TestCase):
    """The photo pipeline's own contract bug, which cost ~600k tokens."""

    def test_library_layout_contract_mentions_required_scene_type(self):
        import library_layout
        self.assertIn('scene_type', library_layout.CONTRACT)
        for name in library_layout.SCHEMA['properties']['scene_type']['enum']:
            self.assertIn(name, library_layout.CONTRACT)


if __name__ == '__main__':
    unittest.main()
