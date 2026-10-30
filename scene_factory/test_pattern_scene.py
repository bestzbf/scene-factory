"""Explicit bounded part-array codec tests; no Blender/API dependency."""
import copy
import json
import math
import unittest
from unittest import mock

import anchored_scene
import compact_scene
import pattern_scene
from model_api import _schema_contract, _walk_wire
from spec_v2 import part
from test_anchored_scene import base_object, bundle, top, wire
from vision_multiview import SCHEMA, response_schema


def make_bundle(groups):
    result = bundle()
    result['shared_scene']['custom_objects'] = groups
    return result


def group(identifier, parts):
    return {'id': identifier, 'position': [0, 0, 0],
            'rotation_degrees': [0, 0, 0], 'material_id': 'wood',
            'confidence': .9, 'observed': True, 'support_id': '', 'parts': parts}


def scene_wire(original):
    raw = wire(original)
    raw[anchored_scene.FORMAT_FIELD] = pattern_scene.FORMAT_VERSION
    for item in raw['shared_scene']['custom_objects']:
        item['part_arrays'] = []
    return raw


def array(raw, group_index=0, part_indices=None, count=None, spacing=None):
    raw['shared_scene']['custom_objects'][group_index]['part_arrays'].append({
        'part_indices': [0] if part_indices is None else part_indices,
        'count': [2, 1, 1] if count is None else count,
        'spacing': [1, 0, 0] if spacing is None else spacing,
    })
    return raw


def expanded_parts(source_parts, specs):
    """Test oracle that explicitly materializes small hand-authored fixtures."""
    by_source = {}
    for array_index, item in enumerate(specs):
        for source_index in item['part_indices']:
            by_source[source_index] = (array_index, item)
    result = []
    for source_index, source in enumerate(source_parts):
        selected = by_source.get(source_index)
        counts = selected[1]['count'] if selected else [1, 1, 1]
        spacing = selected[1]['spacing'] if selected else [0, 0, 0]
        for i in range(counts[0]):
            for j in range(counts[1]):
                for k in range(counts[2]):
                    item = copy.deepcopy(source)
                    item['position'] = [position + index * step for position, index, step
                                        in zip(source['position'], [i, j, k], spacing)]
                    result.append(item)
    return result


class PatternSceneTests(unittest.TestCase):
    def test_ceiling_grid_and_shelf_match_explicit_geometry_and_trace_each_copy(self):
        # Two ceiling beam prototypes make a 4-by-3 lattice; two shelf
        # prototypes make four shelves and two uprights. Only prototypes and
        # explicit translations cross the wire, never inferred geometry.
        ceiling_parts = [
            part('box', size=[4, .05, .03], position=[-2, .4, 2.7], material_id='wood'),
            part('box', size=[.05, 3, .03], position=[-1.2, 0, 2.7],
                 rotation_degrees=[0, 0, 90], material_id='wood'),
        ]
        shelf_parts = [
            part('box', size=[.8, .3, .04], position=[0, 0, .2], material_id='wood'),
            part('box', size=[.04, .3, 1.2], position=[-.38, 0, .6], material_id='wood'),
        ]
        original = make_bundle([group('ceiling_grid', ceiling_parts),
                                group('shelf_frame', shelf_parts)])
        raw = scene_wire(original)
        arrays = [
            [{'part_indices': [0], 'count': [1, 4, 1], 'spacing': [0, .8, 0]},
             {'part_indices': [1], 'count': [3, 1, 1], 'spacing': [1.2, 0, 0]}],
            [{'part_indices': [0], 'count': [1, 1, 4], 'spacing': [0, 0, .3]},
             {'part_indices': [1], 'count': [2, 1, 1], 'spacing': [.76, 0, 0]}],
        ]
        for entry, specs in zip(raw['shared_scene']['custom_objects'], arrays):
            entry['part_arrays'] = copy.deepcopy(specs)
        expected = copy.deepcopy(original)
        for entry, specs in zip(expected['shared_scene']['custom_objects'], arrays):
            entry['parts'] = expanded_parts(entry['parts'], specs)

        raw_before = copy.deepcopy(raw)
        expanded = pattern_scene.expand_response(raw)
        self.assertEqual(expanded, expected)
        self.assertEqual(raw, raw_before)
        self.assertEqual(len(raw['shared_scene']['custom_objects'][0]['parts']), 2)
        self.assertEqual(len(expanded['shared_scene']['custom_objects'][0]['parts']), 7)
        self.assertEqual(len(raw['shared_scene']['custom_objects'][1]['parts']), 2)
        self.assertEqual(len(expanded['shared_scene']['custom_objects'][1]['parts']), 6)
        # Array offsets are added in group local axes before part rotation.
        beams = expanded['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual([item['position'] for item in beams[:4]],
                         [[-2, .4 + .8 * i, 2.7] for i in range(4)])
        self.assertEqual(beams[4]['position'], [-1.2, 0, 2.7])
        self.assertEqual(beams[5]['position'], [0, 0, 2.7])
        self.assertEqual(beams[6]['position'], [1.2, 0, 2.7])
        self.assertEqual([item['rotation_degrees'] for item in beams[4:]],
                         [[0, 0, 90]] * 3)
        receipt = pattern_scene.pattern_report(raw, expanded)
        self.assertEqual(receipt['counts']['source_parts'], 4)
        self.assertEqual(receipt['counts']['expanded_parts'], 13)
        encoded_geometry = json.dumps(
            [{'parts': item['parts'], 'part_arrays': item['part_arrays']}
             for item in raw['shared_scene']['custom_objects']],
            separators=(',', ':'))
        typed_explicit = compact_scene.compact_response(expanded)
        explicit_geometry = json.dumps(
            [{'parts': item['parts']} for item in typed_explicit['shared_scene']['custom_objects']],
            separators=(',', ':'))
        self.assertLess(len(encoded_geometry), len(explicit_geometry))
        ceiling_evidence = receipt['groups'][0]['parts']
        self.assertEqual([(x['source_part_index'], x['repeat_index']) for x in ceiling_evidence],
                         [(0, [0, 0, 0]), (0, [0, 1, 0]), (0, [0, 2, 0]),
                          (0, [0, 3, 0]), (1, [0, 0, 0]), (1, [1, 0, 0]),
                          (1, [2, 0, 0])])
        self.assertEqual(receipt['groups'][0]['part_array_count'], 2)
        converted, _ = pattern_scene._anchored_wire(raw)
        self.assertEqual(pattern_scene.placement_report(raw, expanded),
                         anchored_scene.placement_report(converted, expanded))

    def test_mesh_and_curve_local_topology_is_not_rewritten(self):
        mesh = part('mesh', position=[1, 2, 3], rotation_degrees=[0, 0, 45],
                    material_id='wood', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    faces=[[0, 1, 2]])
        curve = part('polyline', position=[-.5, 0, 1], radius=.02,
                     rotation_degrees=[10, 20, 30], material_id='wood',
                     points=[[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        original = make_bundle([group('geometry', [mesh, curve])])
        raw = array(scene_wire(original), part_indices=[0, 1], count=[2, 1, 1], spacing=[1, 0, 0])
        decoded = pattern_scene.expand_response(raw)['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual(decoded[0]['vertices'], mesh['vertices'])
        self.assertEqual(decoded[1]['vertices'], mesh['vertices'])
        self.assertEqual(decoded[0]['faces'], mesh['faces'])
        self.assertEqual(decoded[1]['faces'], mesh['faces'])
        self.assertEqual(decoded[2]['points'], curve['points'])
        self.assertEqual(decoded[3]['points'], curve['points'])
        self.assertEqual(decoded[0]['position'], [1, 2, 3])
        self.assertEqual(decoded[1]['position'], [2, 2, 3])
        self.assertEqual(decoded[0]['rotation_degrees'], [0, 0, 45])
        self.assertEqual(decoded[2]['position'], [-.5, 0, 1])
        self.assertEqual(decoded[3]['position'], [.5, 0, 1])
        self.assertEqual(decoded[2]['rotation_degrees'], [10, 20, 30])

    def test_three_axis_order_and_unselected_part_preservation(self):
        source = [part(size=[1, 1, 1], position=[1, 2, 3], material_id='wood'),
                  part(size=[.2, .4, .6], position=[-1, -2, -3], material_id='wood'),
                  part('sphere', radius=.2, position=[4, 5, 6], material_id='wood')]
        raw = array(scene_wire(make_bundle([group('three_axes', source)])),
                    part_indices=[2, 0], count=[2, 2, 2], spacing=[1, -2, .5])
        decoded = pattern_scene.expand_response(raw)
        parts = decoded['shared_scene']['custom_objects'][0]['parts']
        expected_indices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.assertEqual([item['position'] for item in parts[:8]],
                         [[1, 2, 3], [1, 2, 3.5], [1, 0, 3], [1, 0, 3.5],
                          [2, 2, 3], [2, 2, 3.5], [2, 0, 3], [2, 0, 3.5]])
        self.assertEqual(parts[8], source[1])
        self.assertEqual(parts[9], source[2])
        self.assertEqual(parts[-1]['position'], [5, 3, 6.5])
        evidence = pattern_scene.pattern_report(raw, decoded)['groups'][0]['parts']
        self.assertEqual([item['repeat_index'] for item in evidence[:8]], expected_indices)
        self.assertEqual(evidence[8], {'part_index': 8, 'source_part_index': 1,
                                       'array_index': None, 'repeat_index': [0, 0, 0]})
        self.assertEqual([item['source_part_index'] for item in evidence], [0] * 8 + [1] + [2] * 8)
        self.assertEqual([item['part_index'] for item in evidence], list(range(17)))

    def test_negative_spacing_and_rotated_group_preserve_all_transforms(self):
        original = make_bundle([group('rotated', [
            part(size=[.2, .1, 1], position=[2, 3, 4],
                 rotation_degrees=[40, 50, 60], material_id='wood')])])
        original['shared_scene']['custom_objects'][0]['position'] = [3, 4, 5]
        original['shared_scene']['custom_objects'][0]['rotation_degrees'] = [10, 20, 90]
        raw = array(scene_wire(original), count=[2, 1, 1], spacing=[-1, 0, 0])
        decoded = pattern_scene.expand_response(raw)['shared_scene']['custom_objects'][0]
        self.assertEqual(decoded['position'], [3, 4, 5])
        self.assertEqual(decoded['rotation_degrees'], [10, 20, 90])
        self.assertEqual([x['position'] for x in decoded['parts']], [[2, 3, 4], [1, 3, 4]])
        self.assertEqual([x['rotation_degrees'] for x in decoded['parts']], [[40, 50, 60]] * 2)

    def test_empty_arrays_and_empty_groups_are_lossless(self):
        original = make_bundle([group('unrepeated', [part(size=[1, 1, 1], material_id='wood')])])
        raw = scene_wire(original)
        self.assertEqual(pattern_scene.expand_response(raw), original)
        report = pattern_scene.pattern_report(raw)
        self.assertEqual(report['counts']['source_parts'], 1)
        self.assertEqual(report['counts']['expanded_parts'], 1)
        self.assertEqual(report['counts']['part_arrays'], 0)
        original['shared_scene']['custom_objects'] = []
        raw = scene_wire(original)
        self.assertEqual(pattern_scene.expand_response(raw), original)
        self.assertEqual(pattern_scene.pattern_report(raw)['groups'], [])

    def test_schema_is_strict_and_requires_array_field(self):
        schema_before = copy.deepcopy(SCHEMA)
        schema = pattern_scene.compact_response_schema(SCHEMA)
        self.assertEqual(SCHEMA, schema_before)
        self.assertEqual(pattern_scene.response_schema(SCHEMA), schema)
        group_rule = schema['properties']['shared_scene']['properties']['custom_objects']['items']
        self.assertIn('part_arrays', group_rule['required'])
        self.assertEqual(schema['properties']['wire_format']['enum'],
                         [pattern_scene.FORMAT_VERSION])
        _schema_contract(schema)
        raw = scene_wire(make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood')])]))
        _walk_wire(raw, schema)
        del raw['shared_scene']['custom_objects'][0]['part_arrays']
        with self.assertRaises(ValueError):
            pattern_scene.expand_response(raw)
        view_schema = response_schema([{'id': 'primary', 'source_image': '/tmp/photo.png',
                                        'width': 640, 'height': 480}], 'stable_id')
        view_compact = pattern_scene.compact_response_schema(view_schema)
        self.assertEqual(view_compact['properties']['scene_id']['enum'], ['stable_id'])
        self.assertEqual(view_compact['properties']['views']['items']['properties']['id']['enum'], ['primary'])

    def test_marker_and_every_array_numeric_boundary_is_strict(self):
        original = make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood')])])
        for marker in ('canonical', 'typed_parts_v1', 'anchored_parts_v1', 'unknown', None):
            raw = scene_wire(original)
            if marker is None:
                del raw['wire_format']
            else:
                raw['wire_format'] = marker
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                pattern_scene.expand_response(raw)
        for field, invalids in (
            ('part_indices', [[], [True], [-1], [100], [0.0], ['0']]),
            ('count', [[0, 1, 1], [101, 1, 1], [2, 1], [2, 1, 1, 1],
                       [True, 1, 1], [math.nan, 1, 1], [10 ** 1000, 1, 1]]),
            ('spacing', [[True, 0, 0], [math.nan, 0, 0], [math.inf, 0, 0],
                         [-math.inf, 0, 0], [10 ** 1000, 0, 0], [51, 0, 0],
                         [1, 0], [1, 0, 0, 0], ['1', 0, 0]]),
        ):
            for invalid in invalids:
                raw = array(scene_wire(original))
                raw['shared_scene']['custom_objects'][0]['part_arrays'][0][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    pattern_scene.expand_response(raw)
        for location in ('part', 'group', 'top'):
            raw = array(scene_wire(original))
            target = (raw if location == 'top' else
                      raw['shared_scene']['custom_objects'][0] if location == 'group' else
                      raw['shared_scene']['custom_objects'][0]['parts'][0])
            target['unexpected'] = 1
            with self.subTest(location=location), self.assertRaises(ValueError):
                pattern_scene.expand_response(raw)

    def test_count_global_part_and_topology_preflight_happens_without_expansion(self):
        base = make_bundle([group('g', [part(size=[.1, .1, .1], material_id='wood')])])
        raw = array(scene_wire(base), count=[100, 100, 100], spacing=[.001, .001, .001])
        # Schema validation creates copies of schemas, but allocation of the
        # raw response or a prototype must never start for this million-copy
        # request. This catches preflight accidentally moved after expansion.
        copied = []
        real_deepcopy = copy.deepcopy
        def track_deepcopy(value, *args, **kwargs):
            copied.append(id(value))
            return real_deepcopy(value, *args, **kwargs)
        with mock.patch.object(pattern_scene.copy, 'deepcopy', side_effect=track_deepcopy):
            with self.assertRaisesRegex(ValueError, 'group exceeds 100 parts'):
                pattern_scene.expand_response(raw)
        self.assertNotIn(id(raw), copied)
        self.assertNotIn(id(raw['shared_scene']['custom_objects'][0]['parts'][0]), copied)

        groups = [group(f'g{i}', [part(size=[.1, .1, .1], material_id='wood')])
                  for i in range(9)]
        raw = scene_wire(make_bundle(groups))
        for entry in raw['shared_scene']['custom_objects']:
            entry['part_arrays'] = [{'part_indices': [0], 'count': [100, 1, 1],
                                     'spacing': [.2, 0, 0]}]
        with self.assertRaisesRegex(ValueError, 'scene exceeds total custom geometry budget'):
            pattern_scene.expand_response(raw)

        vertices = [[i / 100, 0, 0] for i in range(1024)]
        vertices[0], vertices[1], vertices[2] = [0, 0, 0], [1, 0, 0], [0, 1, 0]
        mesh = part('mesh', material_id='wood', vertices=vertices, faces=[[0, 1, 2]])
        raw = array(scene_wire(make_bundle([group('large_mesh', [mesh])])), count=[100, 1, 1],
                    spacing=[.001, 0, 0])
        with self.assertRaisesRegex(ValueError, 'topology exceeds total custom geometry budget'):
            pattern_scene.expand_response(raw)

    def test_exact_group_and_scene_part_budgets_are_allowed(self):
        groups = [group(f'g{i}', [part(size=[.1, .1, .1], material_id='wood')]) for i in range(8)]
        raw = scene_wire(make_bundle(groups))
        for entry in raw['shared_scene']['custom_objects']:
            entry['part_arrays'] = [{'part_indices': [0], 'count': [100, 1, 1],
                                     'spacing': [.1, 0, 0]}]
        expanded = pattern_scene.expand_response(raw)
        self.assertEqual([len(g['parts']) for g in expanded['shared_scene']['custom_objects']], [100] * 8)
        self.assertEqual(pattern_scene.pattern_report(raw, expanded)['counts']['expanded_parts'], 800)
        raw['shared_scene']['custom_objects'][0]['parts'].append(copy.deepcopy(
            raw['shared_scene']['custom_objects'][0]['parts'][0]))
        with self.assertRaisesRegex(ValueError, 'group exceeds 100 parts'):
            pattern_scene.expand_response(raw)

    def test_array_topology_budgets_count_mesh_faces_curves_and_generated_cloth(self):
        cases = [
            part('mesh', material_id='wood', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                 faces=[[0, 1, 2] for _ in range(1001)]),
            part('polyline', material_id='wood', radius=.01,
                 points=[[i / 10, 0, 0] for i in range(161)]),
            part('curtain', material_id='wood', size=[1, .5, 2]),
        ]
        for item in cases:
            raw = array(scene_wire(make_bundle([group('topology', [item])])),
                        count=[100, 1, 1], spacing=[.001, 0, 0])
            with self.subTest(kind=item['type']), self.assertRaisesRegex(
                    ValueError, 'topology exceeds total custom geometry budget'):
                pattern_scene.expand_response(raw)

    def test_topology_receipt_counts_match_materialized_canonical_parts(self):
        parts = [part('mesh', material_id='wood', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                      faces=[[0, 1, 2]]),
                 part('curve', material_id='wood', radius=.01, points=[[0, 0, 0], [1, 0, 0]]),
                 part('curtain', material_id='wood', size=[1, .5, 2])]
        raw = array(scene_wire(make_bundle([group('mixed', parts)])), part_indices=[0, 1, 2])
        counts = pattern_scene.pattern_report(raw)['counts']
        self.assertEqual(counts['expanded_parts'], 6)
        self.assertEqual(counts['vertices'], 6 + 2 * 65 * 25 * 2)
        self.assertEqual(counts['face_indices'], 6 + 2 * (64 * 24 * 8 + (64 + 24) * 8))
        self.assertEqual(counts['curve_points'], 4)

    def test_canonical_geometry_validation_still_rejects_invalid_mesh_and_curve(self):
        original = make_bundle([group('bad_geometry', [part(size=[1, 1, 1], material_id='wood')])])
        cases = [
            {'type': 'mesh', 'position': [0, 0, 0], 'rotation_degrees': [0, 0, 0],
             'material_id': 'wood', 'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
             'faces': [[0, 1, 3]]},
            {'type': 'mesh', 'position': [0, 0, 0], 'rotation_degrees': [0, 0, 0],
             'material_id': 'wood', 'vertices': [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
             'faces': [[0, 1, 2]]},
            {'type': 'curve', 'position': [0, 0, 0], 'rotation_degrees': [0, 0, 0],
             'material_id': 'wood', 'points': [[0, 0, 0], [0, 0, 0]], 'radius': .1},
        ]
        for invalid in cases:
            raw = array(scene_wire(original))
            raw['shared_scene']['custom_objects'][0]['parts'] = [invalid]
            with self.subTest(kind=invalid['type']), self.assertRaises(ValueError):
                pattern_scene.expand_response(raw)

    def test_arrays_are_unique_and_reject_bad_indices_counts_offsets_and_fields(self):
        original = make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood'),
                                            part(size=[1, 1, 1], material_id='wood')])])
        mutations = []
        raw = scene_wire(original)
        raw['shared_scene']['custom_objects'][0]['part_arrays'] = [
            {'part_indices': [0], 'count': [2, 1, 1], 'spacing': [1, 0, 0]},
            {'part_indices': [0], 'count': [1, 2, 1], 'spacing': [0, 1, 0]}]
        mutations.append(raw)
        raw = scene_wire(original)
        raw['shared_scene']['custom_objects'][0]['part_arrays'] = [
            {'part_indices': [0, 0], 'count': [2, 1, 1], 'spacing': [1, 0, 0]}]
        mutations.append(raw)
        for bad_count in ([1, 1, 1], [101, 1, 1], [True, 1, 1], [2.0, 1, 1]):
            raw = array(scene_wire(original), count=bad_count, spacing=[1, 0, 0])
            mutations.append(raw)
        for bad in (
            lambda a: a.update(part_indices=[2]),
            lambda a: a.update(count=[2, 1, 1], spacing=[0, 0, 0]),
            lambda a: a.update(count=[2, 1, 1], spacing=[math.nan, 0, 0]),
            lambda a: a.update(unexpected=True),
        ):
            raw = scene_wire(original)
            item = {'part_indices': [0], 'count': [2, 1, 1], 'spacing': [1, 0, 0]}
            bad(item)
            raw['shared_scene']['custom_objects'][0]['part_arrays'] = [item]
            mutations.append(raw)
        for raw in mutations:
            with self.subTest(part_arrays=raw['shared_scene']['custom_objects'][0]['part_arrays']), self.assertRaises(ValueError):
                pattern_scene.expand_response(raw)

    def test_rounding_collapsed_offsets_fail_per_prototype_but_distinct_prototypes_may_coincide(self):
        source = part(size=[1, 1, 1], position=[1, 2, 3], material_id='wood')
        original = make_bundle([group('g', [source])])
        for axis in range(3):
            counts, spacing = [1, 1, 1], [0, 0, 0]
            counts[axis], spacing[axis] = 2, 1e-20
            raw = array(scene_wire(original), count=counts, spacing=spacing)
            before = copy.deepcopy(raw)
            real_copy = copy.deepcopy
            def no_raw_copy(value, *args, **kwargs):
                if value is raw:
                    raise AssertionError('Collapsed positions must fail before output allocation')
                return real_copy(value, *args, **kwargs)
            with self.subTest(axis=axis), mock.patch.object(pattern_scene.copy, 'deepcopy', side_effect=no_raw_copy):
                with self.assertRaisesRegex(ValueError, 'produces duplicate expanded positions'):
                    pattern_scene.expand_response(raw)
            self.assertEqual(raw, before)

        # The same array can be representable for a zero-origin prototype yet
        # collapse at another prototype's nonzero position. Check each source.
        original = make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood'), source])])
        raw = array(scene_wire(original), part_indices=[0, 1], count=[2, 1, 1], spacing=[1e-20, 0, 0])
        with self.assertRaisesRegex(ValueError, 'produces duplicate expanded positions'):
            pattern_scene.expand_response(raw)

        # Do not prohibit intentional coincident prototypes. Each prototype's
        # own array is unique; collision policy belongs to scene geometry and
        # can represent layered/coincident parts explicitly.
        original = make_bundle([group('g', [copy.deepcopy(source), copy.deepcopy(source)])])
        raw = scene_wire(original)
        raw['shared_scene']['custom_objects'][0]['part_arrays'] = [
            {'part_indices': [0], 'count': [2, 1, 1], 'spacing': [1, 0, 0]},
            {'part_indices': [1], 'count': [2, 1, 1], 'spacing': [1, 0, 0]},
        ]
        parts = pattern_scene.expand_response(raw)['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual([item['position'] for item in parts],
                         [[1, 2, 3], [2, 2, 3], [1, 2, 3], [2, 2, 3]])

    def test_out_of_bounds_expanded_position_and_unknown_material_are_rejected(self):
        original = make_bundle([group('g', [part(size=[1, 1, 1], position=[50, 0, 0],
                                                material_id='wood')])])
        raw = array(scene_wire(original), count=[2, 1, 1], spacing=[1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            pattern_scene.expand_response(raw)
        raw = scene_wire(original)
        raw['shared_scene']['custom_objects'][0]['parts'][0]['material_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unknown reference'):
            pattern_scene.expand_response(raw)

    def test_invalid_base_support_is_rejected_through_anchored_decoder(self):
        raw = array(scene_wire(make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood')])])) )
        raw['shared_scene']['base_scene']['objects'][0]['placement'] = {
            'mode': 'on_top', 'xy': [0, 2], 'support_id': 'missing', 'clearance': 0}
        with self.assertRaisesRegex(ValueError, 'existing base support'):
            pattern_scene.expand_response(raw)

    def test_patterns_preserve_corresponding_base_repeat_support_evidence(self):
        original = make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood')])])
        original['shared_scene']['base_scene']['objects'] = [
            base_object('desks', repeat=[1, 3], spacing=[0, 1.8]),
            base_object('screens', 'monitor', repeat=[1, 3], spacing=[0, 1.8])]
        raw = array(scene_wire(original))
        raw['shared_scene']['base_scene']['objects'][1]['placement'] = top('desks__0_0')
        before = copy.deepcopy(raw)
        expanded = pattern_scene.expand_response(raw)
        self.assertEqual(expanded['shared_scene']['base_scene']['objects'][1]['position'],
                         [0, 2, .74])
        evidence = pattern_scene.placement_report(raw, expanded)['instances'][3:]
        self.assertEqual([x['support_id'] for x in evidence],
                         ['desks__0_0', 'desks__0_1', 'desks__0_2'])
        self.assertTrue(all(x['support_binding'] == 'matching_repeat_index' for x in evidence))
        self.assertEqual(raw, before)

    def test_report_rejects_tampered_expansion_and_raw_is_never_mutated(self):
        raw = array(scene_wire(make_bundle([group('g', [part(size=[1, 1, 1], material_id='wood')])])) )
        before = copy.deepcopy(raw)
        expanded = pattern_scene.expand_response(raw)
        pattern_scene.pattern_report(raw, expanded)
        self.assertEqual(raw, before)
        report = pattern_scene.pattern_report(raw, expanded)
        report['groups'][0]['parts'][0]['repeat_index'][0] = 99
        self.assertEqual(pattern_scene.pattern_report(raw, expanded)['groups'][0]['parts'][0]['repeat_index'], [0, 0, 0])
        self.assertEqual(raw, before)
        self.assertEqual(expanded['shared_scene']['custom_objects'][0]['parts'][0]['position'], [0, 0, 0])
        expanded['shared_scene']['custom_objects'][0]['parts'][1]['position'][0] = 99
        with self.assertRaisesRegex(ValueError, 'does not match'):
            pattern_scene.pattern_report(raw, expanded)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            pattern_scene.placement_report(raw, expanded)
        self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
