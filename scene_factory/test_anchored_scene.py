"""Explicit placement decoding; no Blender, credentials or network required."""
import copy
import math
import unittest

import compact_scene
from anchored_scene import (CONTACT_TOLERANCE_METRES, FORMAT_FIELD, FORMAT_VERSION,
                            compact_response_schema, expand_response, placement_report,
                            response_schema)
from model_api import _schema_contract, _walk_wire
from spec_v2 import part
from test_multiview import fixture
from test_spec_v2 import fixture as scene_fixture
from vision_multiview import SCHEMA, response_schema as full_response_schema


def base_object(identifier='desk', category='desk', position=None, dimensions=None,
                rotation=0, repeat=None, spacing=None):
    return {'id': identifier, 'category': category,
            'position': [0, 2, 0] if position is None else position,
            'dimensions': [2, 1, .74] if dimensions is None else dimensions,
            'rotation_z_deg': rotation, 'color': [.4, .3, .2],
            'repeat': [1, 1] if repeat is None else repeat,
            'spacing': [0, 0] if spacing is None else spacing, 'confidence': .8}


def bundle(objects=None):
    result = fixture()
    result['shared_scene'] = scene_fixture()
    result['shared_scene']['base_scene']['objects'] = objects or [base_object()]
    result['shared_scene'].setdefault('base_material_overrides', [])
    result['shared_scene'].setdefault('exterior_contexts', [])
    return result


def wire(original=None, placements=None):
    """Fixture encoder is explicit and defaults to lossless world placement."""
    encoded = compact_scene.compact_response(original or bundle())
    encoded[FORMAT_FIELD] = FORMAT_VERSION
    for item in encoded['shared_scene']['base_scene']['objects']:
        position = item.pop('position')
        item['placement'] = copy.deepcopy((placements or {}).get(
            item['id'], {'mode': 'world', 'position': position}))
    return encoded


def top(support='desk', xy=None, clearance=0):
    return {'mode': 'on_top', 'xy': [0, 2] if xy is None else xy,
            'support_id': support, 'clearance': clearance}


class AnchoredSceneTests(unittest.TestCase):
    def test_floor_desk_tabletop_height_and_monitor_contact(self):
        original = bundle([base_object(position=[0, 2, .72]),
                           base_object('screen', 'monitor', dimensions=[.6, .2, .45])])
        raw = wire(original, {'desk': {'mode': 'floor', 'xy': [0, 2]},
                              'screen': top(clearance=.01)})
        before = copy.deepcopy(raw)
        expanded = expand_response(raw)
        desk, screen = expanded['shared_scene']['base_scene']['objects']
        self.assertEqual(desk['position'], [0, 2, 0])
        self.assertEqual(desk['position'][2] + desk['dimensions'][2], .74)
        self.assertEqual(screen['position'], [0, 2, .75])
        self.assertNotIn('placement', screen)
        self.assertNotIn(FORMAT_FIELD, expanded)
        self.assertEqual(raw, before)
        receipt = placement_report(raw, expanded)
        self.assertEqual(receipt['format'], FORMAT_VERSION)
        self.assertEqual(receipt['instances'][1]['support_top_z'], .74)
        self.assertEqual(receipt['instances'][1]['resolved_bottom_position'], [0, 2, .75])

    def test_world_preserves_legitimately_raised_wall_furniture_and_unrelated_data(self):
        original = bundle([base_object('raised', 'cabinet', position=[1, 3, 1.2]),
                           base_object('mounted', 'monitor', position=[-1, 4, 1.4])])
        before = copy.deepcopy(original)
        raw = wire(original)
        self.assertEqual(expand_response(raw), original)
        self.assertEqual(original, before)
        self.assertEqual(raw['shared_scene']['base_scene']['objects'][0]['placement']['position'][2], 1.2)
        report = placement_report(raw)
        self.assertEqual(report['instances'][0]['support_id'], '')
        self.assertIsNone(report['instances'][0]['support_top_z'])

    def test_support_graph_resolves_out_of_order_nested_contacts(self):
        objects = [base_object('screen', 'monitor', dimensions=[.5, .2, .4]),
                   base_object('upper', 'box', dimensions=[1, .6, .3]),
                   base_object('lower', 'cabinet', dimensions=[2, 1, 1])]
        raw = wire(bundle(objects), {'screen': top('upper', clearance=.02),
                                    'upper': top('lower', clearance=.01),
                                    'lower': {'mode': 'floor', 'xy': [0, 2]}})
        expanded = expand_response(raw)
        positions = {item['id']: item['position'] for item in expanded['shared_scene']['base_scene']['objects']}
        self.assertEqual(positions['lower'], [0, 2, 0])
        self.assertAlmostEqual(positions['upper'][2], 1.01)
        self.assertAlmostEqual(positions['screen'][2], 1.33)

    def test_exact_repeated_parent_instance_and_deterministic_instance_evidence(self):
        objects = [base_object(repeat=[2, 2], spacing=[3, 2]),
                   base_object('screen', 'monitor', dimensions=[.5, .2, .4])]
        raw = wire(bundle(objects), {'desk': {'mode': 'floor', 'xy': [-2, 2]},
                                    'screen': top('desk__1_1', xy=[1, 4])})
        expanded = expand_response(raw)
        self.assertEqual(expanded['shared_scene']['base_scene']['objects'][1]['position'], [1, 4, .74])
        receipt = placement_report(raw)
        self.assertEqual([item['object_id'] for item in receipt['instances']],
                         ['desk__0_0', 'desk__0_1', 'desk__1_0', 'desk__1_1', 'screen'])
        self.assertEqual(receipt['instances'][3]['resolved_bottom_position'], [1, 4, 0])
        self.assertEqual(receipt, placement_report(raw))
        receipt['instances'][0]['placement']['xy'][0] = 100
        self.assertEqual(raw['shared_scene']['base_scene']['objects'][0]['placement']['xy'], [-2, 2])

    def test_repeated_on_top_children_must_all_fit_one_exact_parent_instance(self):
        objects = [base_object(dimensions=[2, 1, .74]),
                   base_object('screens', 'monitor', dimensions=[.5, .1, .4],
                               repeat=[3, 1], spacing=[.5, 0])]
        raw = wire(bundle(objects), {'desk': {'mode': 'floor', 'xy': [0, 2]},
                                     'screens': top('desk', xy=[-.5, 2])})
        result = expand_response(raw)
        screens = result['shared_scene']['base_scene']['objects'][1]
        self.assertEqual(screens['position'], [-.5, 2, .74])
        self.assertEqual([[-.5 + i * .5, 2, .74] for i in range(3)],
                         [[screens['position'][0] + i * screens['spacing'][0],
                           screens['position'][1] + i * screens['spacing'][1],
                           screens['position'][2]] for i in range(3)])
        objects[1]['spacing'] = [1.1, 0]
        raw = wire(bundle(objects), {'desk': {'mode': 'floor', 'xy': [0, 2]},
                                     'screens': top('desk', xy=[-.5, 2])})
        with self.assertRaisesRegex(ValueError, 'footprint'):
            expand_response(raw)

    def test_identical_repeat_grids_bind_corresponding_supports_without_relocating_xy(self):
        objects = [base_object(repeat=[2, 2], spacing=[-3, 2], rotation=90),
                   base_object('screens', 'monitor', dimensions=[.5, .1, .4],
                               repeat=[2, 2], spacing=[-3, 2])]
        raw = wire(bundle(objects), {'desk': {'mode': 'floor', 'xy': [2, 2]},
                                    'screens': top('desk__0_0', xy=[2.125, 2.25], clearance=.01)})
        before = copy.deepcopy(raw)
        expanded = expand_response(raw)
        screens = expanded['shared_scene']['base_scene']['objects'][1]
        self.assertEqual(screens['position'], [2.125, 2.25, .75])
        self.assertEqual(screens['repeat'], [2, 2])
        self.assertEqual(screens['spacing'], [-3, 2])
        evidence = placement_report(raw, expanded)['instances'][4:]
        self.assertEqual([x['support_id'] for x in evidence],
                         ['desk__0_0', 'desk__0_1', 'desk__1_0', 'desk__1_1'])
        self.assertEqual([x['support_binding'] for x in evidence],
                         ['matching_repeat_index'] * 4)
        self.assertEqual([x['resolved_bottom_position'] for x in evidence],
                         [[2.125, 2.25, .75], [2.125, 4.25, .75],
                          [-.875, 2.25, .75], [-.875, 4.25, .75]])
        self.assertTrue(all(x['placement']['support_id'] == 'desk__0_0' for x in evidence))
        self.assertEqual(raw, before)

    def test_corresponding_repeat_supports_resolve_nested_contacts_out_of_order(self):
        repeated = {'repeat': [1, 3], 'spacing': [0, 2]}
        objects = [base_object('screens', 'monitor', dimensions=[.5, .1, .4], **repeated),
                   base_object('boxes', 'box', dimensions=[1, .5, .2], **repeated),
                   base_object('desks', 'desk', **repeated)]
        raw = wire(bundle(objects), {'screens': top('boxes__0_0'),
                                    'boxes': top('desks__0_0'),
                                    'desks': {'mode': 'floor', 'xy': [0, 2]}})
        expanded = expand_response(raw)
        self.assertEqual(expanded['shared_scene']['base_scene']['objects'][0]['position'],
                         [0, 2, .94])
        evidence = placement_report(raw)['instances']
        self.assertEqual([x['support_id'] for x in evidence[:3]],
                         ['boxes__0_0', 'boxes__0_1', 'boxes__0_2'])
        self.assertEqual([x['support_id'] for x in evidence[3:6]],
                         ['desks__0_0', 'desks__0_1', 'desks__0_2'])

    def test_repeat_binding_requires_exact_grid_and_origin_support_and_still_checks_contact(self):
        objects = [base_object(repeat=[1, 3], spacing=[0, 2]),
                   base_object('screens', 'monitor', repeat=[1, 3], spacing=[0, 2])]
        raw = wire(bundle(objects), {'desk': {'mode': 'floor', 'xy': [0, 2]},
                                    'screens': top('desk__0_0')})
        for field, value in (('repeat', [1, 2]), ('spacing', [0, 2.000000001]),
                             ('spacing', [0, -2])):
            invalid = copy.deepcopy(raw)
            invalid['shared_scene']['base_scene']['objects'][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'footprint'):
                expand_response(invalid)
        invalid = copy.deepcopy(raw)
        invalid['shared_scene']['base_scene']['objects'][1]['placement']['support_id'] = 'desk__0_1'
        with self.assertRaisesRegex(ValueError, 'screens__0_0 on support desk__0_1.*footprint'):
            expand_response(invalid)
        invalid = copy.deepcopy(raw)
        invalid['shared_scene']['base_scene']['objects'][1]['placement']['xy'][0] = 2
        before = copy.deepcopy(invalid)
        with self.assertRaisesRegex(ValueError, 'screens__0_0 on support desk__0_0.*footprint'):
            expand_response(invalid)
        self.assertEqual(invalid, before)

    def test_repeated_parent_root_alias_and_out_of_range_instance_rejected(self):
        objects = [base_object(repeat=[2, 1], spacing=[3, 0]),
                   base_object('screen', 'monitor')]
        for support in ('desk', 'desk__2_0', 'missing', '', 'custom_shelf', 'floor'):
            raw = wire(bundle(objects), {'screen': top(support)})
            with self.subTest(support=support), self.assertRaises(ValueError):
                expand_response(raw)

    def test_cycle_self_support_and_repeated_child_is_checked(self):
        objects = [base_object('a', 'table'), base_object('b', 'box')]
        for placements in ({'a': top('a')}, {'a': top('b'), 'b': top('a')}):
            with self.subTest(placements=placements), self.assertRaisesRegex(ValueError, 'Cyclic'):
                expand_response(wire(bundle(objects), placements))
        objects[1]['repeat'] = [2, 1]
        objects[1]['spacing'] = [1, 0]
        expanded = expand_response(wire(bundle(objects), {'b': top('a')}))
        self.assertEqual(expanded['shared_scene']['base_scene']['objects'][1]['position'], [0, 2, .74])
        objects[1]['spacing'] = [3, 0]
        with self.assertRaisesRegex(ValueError, 'footprint'):
            expand_response(wire(bundle(objects), {'b': top('a')}))

    def test_only_flat_outer_top_categories_can_be_supports(self):
        for category in ('chair', 'sofa', 'bed', 'plant', 'monitor', 'linear_light'):
            original = bundle([base_object('parent', category), base_object('child', 'box')])
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, 'flat outer top'):
                expand_response(wire(original, {'child': top('parent')}))
        for category in ('desk', 'table', 'cabinet', 'shelf', 'box'):
            original = bundle([base_object('parent', category), base_object('child', 'box')])
            with self.subTest(category=category):
                self.assertEqual(expand_response(wire(original, {'child': top('parent')}))
                                 ['shared_scene']['base_scene']['objects'][1]['position'][2], .74)

    def test_yaw_rotated_contact_footprint_and_bounded_tolerance(self):
        objects = [base_object(dimensions=[2, .4, .74], rotation=90),
                   base_object('screen', 'monitor', dimensions=[4, 4, .4])]
        # Bottom-centre contact is tested; a wide screen may overhang the top.
        for xy in ([.15, 2.9], [.2 + CONTACT_TOLERANCE_METRES, 2]):
            with self.subTest(xy=xy):
                expand_response(wire(bundle(objects), {'screen': top(xy=xy)}))
        for xy in ([.23, 2], [0, 3.03]):
            with self.subTest(xy=xy), self.assertRaisesRegex(ValueError, 'footprint'):
                expand_response(wire(bundle(objects), {'screen': top(xy=xy)}))

    def test_floor_requires_canonical_floor_but_world_can_use_custom_raised_floor(self):
        original = bundle()
        original['architecture']['omit_surfaces'].append('floor')
        with self.assertRaisesRegex(ValueError, 'canonical room floor'):
            expand_response(wire(original, {'desk': {'mode': 'floor', 'xy': [0, 2]}}))
        original['shared_scene']['base_scene']['objects'][0]['position'][2] = .3
        self.assertEqual(expand_response(wire(original)), original)

    def test_explicit_markers_required_and_fields_cannot_be_smuggled_between_branches(self):
        for marker in ('typed_parts_v1', 'unknown', None):
            raw = wire()
            if marker is None:
                del raw[FORMAT_FIELD]
            else:
                raw[FORMAT_FIELD] = marker
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                expand_response(raw)
        mutations = [
            lambda item: item.update(position=[0, 2, 0]),
            lambda item: item['placement'].update(extra=0),
            lambda item: item.update(placement={'mode': 'floor', 'xy': [0, 2], 'position': [0, 2, 0]}),
            lambda item: item.update(placement={'mode': 'on_top', 'xy': [0, 2], 'support_id': 'desk'}),
            lambda item: item.update(placement={'mode': 'floor', 'xy': [0, 2, 0]}),
            lambda item: item.update(placement={'mode': 'centre', 'position': [0, 2, .37]}),
        ]
        for mutate in mutations:
            raw = wire()
            mutate(raw['shared_scene']['base_scene']['objects'][0])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                expand_response(raw)

    def test_nonfinite_bool_coordinates_and_clearance_outside_range_fail(self):
        for bad in (math.nan, math.inf, -math.inf, True, '0', 51, 10 ** 1000):
            raw = wire(None, {'desk': {'mode': 'floor', 'xy': [bad, 2]}})
            with self.subTest(coordinate=bad), self.assertRaises(ValueError):
                expand_response(raw)
        original = bundle([base_object(), base_object('child', 'box')])
        for bad in (-.001, .0301, math.nan, math.inf, True):
            with self.subTest(clearance=bad), self.assertRaises(ValueError):
                expand_response(wire(original, {'child': top(clearance=bad)}))
        expand_response(wire(original, {'child': top(clearance=.03)}))

    def test_resolved_height_and_every_repeated_instance_stay_in_canonical_bounds(self):
        original = bundle([base_object(position=[0, 2, 49], dimensions=[1, 1, 2]),
                           base_object('child', 'box')])
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            expand_response(wire(original, {'child': top()}))
        original = bundle([base_object(position=[49, 2, 0], repeat=[2, 1], spacing=[2, 0])])
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            expand_response(wire(original))

    def test_duplicate_coincident_repeats_and_namespace_collisions_fail(self):
        originals = [bundle([base_object(), base_object()]),
                     bundle([base_object(repeat=[2, 1], spacing=[0, 0])]),
                     bundle([base_object(repeat=[2, 1], spacing=[3, 0]),
                             base_object('desk__0_0', repeat=[2, 1], spacing=[0, 3])])]
        # Build wire directly: canonical compaction correctly rejects some of
        # these before they could reach an encoder in the first place.
        for original in originals:
            raw = wire()
            raw['shared_scene']['base_scene']['objects'] = copy.deepcopy(original['shared_scene']['base_scene']['objects'])
            for item in raw['shared_scene']['base_scene']['objects']:
                item['placement'] = {'mode': 'world', 'position': item.pop('position')}
            with self.subTest(objects=original['shared_scene']['base_scene']['objects']), self.assertRaises(ValueError):
                expand_response(raw)
        for namespace in ('custom_objects', 'lights'):
            raw = wire()
            raw['shared_scene'][namespace][0]['id'] = 'desk'
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                expand_response(raw)

    def test_typed_part_geometry_checks_are_unchanged(self):
        original = bundle()
        original['shared_scene']['custom_objects'][0]['parts'] = [
            part('mesh', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])]
        raw = wire(original)
        self.assertEqual(expand_response(raw), original)
        raw['shared_scene']['custom_objects'][0]['parts'][0]['faces'] = []
        with self.assertRaises(ValueError):
            expand_response(raw)

    def test_schema_retains_identity_restrictions_is_strict_and_does_not_mutate(self):
        views = [{'id': 'exact', 'source_image': '/trusted/photo.jpg', 'width': 1234, 'height': 567}]
        full = full_response_schema(views, 'immutable_scene')
        before = copy.deepcopy(full)
        encoded = compact_response_schema(full)
        self.assertEqual(full, before)
        self.assertEqual(response_schema(full), encoded)
        self.assertEqual(encoded['properties']['views'], full['properties']['views'])
        self.assertEqual(encoded['properties']['scene_id'], full['properties']['scene_id'])
        properties = encoded['properties']['shared_scene']['properties']['base_scene']['properties']['objects']['items']['properties']
        self.assertNotIn('position', properties)
        self.assertEqual(len(properties['placement']['anyOf']), 3)
        _schema_contract(encoded)
        _walk_wire(wire(), compact_response_schema(SCHEMA))
        bad = copy.deepcopy(SCHEMA)
        bad['properties']['shared_scene']['properties']['base_scene']['properties']['objects']['items']['properties']['position']['maxItems'] = 2
        with self.assertRaises(ValueError):
            compact_response_schema(bad)

    def test_placement_receipt_rejects_unrelated_or_tampered_expansion(self):
        raw = wire()
        expanded = expand_response(raw)
        before_raw, before_expanded = copy.deepcopy(raw), copy.deepcopy(expanded)
        placement_report(raw, expanded)
        self.assertEqual(raw, before_raw)
        self.assertEqual(expanded, before_expanded)
        expanded['shared_scene']['base_scene']['objects'][0]['position'][2] = 3
        with self.assertRaisesRegex(ValueError, 'does not match'):
            placement_report(raw, expanded)


if __name__ == '__main__':
    unittest.main()
