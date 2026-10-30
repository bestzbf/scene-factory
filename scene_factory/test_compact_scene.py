"""Typed-parts codec tests: saved scenes, geometry failures and identity safety."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compact_scene import (COMMON, FIELDS_BY_TYPE, FIELD_VARIANTS, FORMAT_FIELD, FORMAT_VERSION,
                           compact_response, compact_response_schema, expand_response)
from multiview_spec import validate
from spec_v2 import PART_SCHEMA, part
from test_multiview import fixture
from test_spec_v2 import fixture as scene_fixture
from vision_multiview import SCHEMA, normalize_response, response_schema
import vision_multiview as vision
import model_api


def bundle_with_parts(parts):
    bundle = fixture()
    bundle['shared_scene'] = scene_fixture()
    first = bundle['views'][0]['camera']
    bundle['shared_scene']['base_scene']['camera'] = {
        key: first[key] for key in ('position', 'target', 'lens_mm')}
    bundle['shared_scene']['camera']['roll_degrees'] = first['roll_degrees']
    bundle['shared_scene']['custom_objects'][0]['parts'] = parts
    return bundle


def all_types():
    return [
        part('box', size=[1, .2, .3]),
        part('sphere', radius=.2),
        part('sphere', size=[.3, .5, .7], radius=.1),
        part('cylinder', size=[.4, .5, 1], radius=.1),
        part('curve', points=[[0, 0, 0], [.5, .2, 1]], radius=.01),
        part('mesh', vertices=[[0, 0, 0], [1, 0, 0], [0, 0, 1]], faces=[[0, 1, 2]]),
        part('image_plane', size=[1, .3, 1], material_id='photo'),
        part('curtain', size=[1, .2, 1.5]),
        part('polyline', points=[[-.4, 0, 0], [-.4, 0, .8], [.4, 0, .8], [.4, 0, 0]], radius=.01),
    ]


def canonical_ignored(bundle):
    result = copy.deepcopy(bundle)
    for group in result['shared_scene']['custom_objects']:
        for item in group['parts']:
            item.setdefault('cloth', [])
    return result


class CompactSceneTests(unittest.TestCase):
    def prepare_analysis(self, directory):
        from PIL import Image
        bundle = bundle_with_parts(all_types())
        for view in bundle['views']:
            image = directory / (view['id'] + '.png')
            Image.new('RGB', (view['width'], view['height']), (40, 60, 80)).save(image)
            view['source_image'] = str(image)
        first = bundle['views'][0]
        bundle['shared_scene']['image'] = {'width': first['width'], 'height': first['height'], 'source_id': first['id']}
        views = [{key: v[key] for key in ('id', 'source_image', 'width', 'height')} for v in bundle['views']]
        raw = copy.deepcopy(bundle)
        for view in raw['views']:
            view['source_image'] = 'input:' + view['id']
        return bundle, views, compact_response(raw)

    def test_analysis_opt_in_expands_before_normalizing_and_records_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            expected, views, wire = self.prepare_analysis(directory)
            output = directory / 'initial_bundle.json'
            def invoke(prompt, images, schema, path, *args):
                vision.atomic_json(path.with_name(path.stem + '.raw.json'), wire)
                return wire
            with patch.object(vision, '_invoke_model', side_effect=invoke) as model:
                result = vision.analyze('room', views, output, wire_format=FORMAT_VERSION)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(result, expected)
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertEqual(json.loads(output.with_name('initial_bundle.raw.json').read_text()), wire)
            self.assertEqual(model.call_args.args[1], [v['source_image'] for v in views])
            self.assertIn('typed_parts_v1', model.call_args.args[0])
            self.assertEqual(model.call_args.args[2], compact_response_schema(response_schema(views, 'room')))
            provenance = json.loads((directory / 'analysis_provenance.json').read_text())
            encoding = provenance['scene_wire_encoding']
            self.assertEqual(encoding['format'], FORMAT_VERSION)
            self.assertLess(encoding['wire_utf8_bytes'], encoding['expanded_utf8_bytes'])
            self.assertEqual(encoding['implementation_sha256'], vision.file_sha256(Path(__file__).with_name('compact_scene.py')))
            self.assertEqual(provenance['normalizations']['supports'], [])

    def test_analysis_rejects_bad_wire_without_fallback_or_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            _, views, wire = self.prepare_analysis(directory)
            for mutation in ('empty_faces', 'bad_view', 'wrong_marker'):
                bad = copy.deepcopy(wire)
                if mutation == 'empty_faces':
                    bad['shared_scene']['custom_objects'][0]['parts'][5]['faces'] = []
                elif mutation == 'bad_view':
                    bad['views'][0]['source_image'] = '/untrusted/new.png'
                else:
                    bad[FORMAT_FIELD] = 'unknown'
                output = directory / mutation / 'initial_bundle.json'
                with patch.object(vision, '_invoke_model', return_value=bad) as model:
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        vision.analyze('room', views, output, wire_format=FORMAT_VERSION)
                self.assertEqual(model.call_count, 1)
                self.assertFalse(output.exists())
            with patch.object(vision, '_invoke_model') as model:
                with self.assertRaises(ValueError):
                    vision.analyze('room', views, directory / 'bad.json', wire_format='unknown')
                model.assert_not_called()

    def test_analysis_compact_cache_cannot_cross_wire_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            expected, views, wire = self.prepare_analysis(directory)
            for key in ('base_material_overrides','exterior_contexts'):
                expected['shared_scene'].setdefault(key,[])
                wire['shared_scene'].setdefault(key,[])
            cache = directory / 'cache' / 'room'
            cache.mkdir(parents=True)
            config={'provider':'openai_chat','model':'fixture-compact-cache',
                    'base_url':'http://127.0.0.1:1/v1','api_key_env':''}
            with patch.object(vision, '_invoke_model', return_value=wire):
                vision.analyze('room', views, cache / 'initial_bundle.json', wire_format=FORMAT_VERSION,model_config=config)
            vision.atomic_json(cache / 'initial_bundle.raw.json', wire)
            prompt=(cache/'initial_bundle.prompt.txt').read_text(encoding='utf-8')
            identity=model_api._request_identity(prompt,[v['source_image'] for v in views],
                compact_response_schema(response_schema(views,'room')),config)
            digest=vision.file_sha256(cache/'initial_bundle.raw.json')
            vision.atomic_json(cache/'initial_bundle.api.json',{**identity,
                'status':'completed','attempts':1,'http_status':200,'finish_reason':'stop',
                'raw_response_sha256':digest,'transport_attempts':[{'phase':'response_complete',
                    'request_started':True,'request_sent':True,'http_status':200,'raw_response_sha256':digest}]})
            vision.atomic_json(cache / 'job.json', {
                'scene_id': 'room', 'pipeline': {}, 'models':{'analysis':config},'sources': [
                    {'path': v['source_image'], 'sha256': vision.file_sha256(v['source_image'])} for v in views]})
            for view in views:
                target = cache / 'sources' / view['id'] / 'reference.png'
                target.parent.mkdir(parents=True)
                target.write_bytes(Path(view['source_image']).read_bytes())
            with patch.object(vision, '_invoke_model') as model:
                reused = vision.analyze('room', views, directory / 'reused' / 'initial_bundle.json',
                                        raw_cache=cache.parent, wire_format=FORMAT_VERSION,model_config=config)
                self.assertEqual(reused, expected)
                model.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'prompt'):
                    vision.analyze('room', views, directory / 'wrong_format' / 'initial_bundle.json', raw_cache=cache.parent,model_config=config)
                model.assert_not_called()

    def test_all_geometry_types_roundtrip_and_mutation_isolation(self):
        original = bundle_with_parts(all_types())
        before = copy.deepcopy(original)
        wire = compact_response(original)
        self.assertEqual(original, before)
        self.assertEqual(wire[FORMAT_FIELD], FORMAT_VERSION)
        for item in wire['shared_scene']['custom_objects'][0]['parts']:
            self.assertIn(set(item), [set(COMMON + fields) for fields in FIELD_VARIANTS[item['type']]])
        before_wire = copy.deepcopy(wire)
        decoded = expand_response(wire)
        self.assertEqual(wire, before_wire)
        self.assertEqual(decoded, original)
        validate(decoded)
        decoded['views'][0]['camera']['position'][0] = 9
        decoded['shared_scene']['custom_objects'][0]['parts'][0]['size'][0] = 9
        self.assertEqual(wire, before_wire)
        self.assertEqual(original, before)

    def test_real_repository_scenes_roundtrip_without_visual_data_loss(self):
        for path in sorted((Path(__file__).parent / 'examples_v2').glob('*.json')):
            with self.subTest(scene=path.name):
                bundle = fixture()
                bundle['shared_scene'] = json.loads(path.read_text())
                first = bundle['views'][0]['camera']
                bundle['shared_scene']['base_scene']['camera'] = {
                    key: first[key] for key in ('position', 'target', 'lens_mm')}
                bundle['shared_scene']['camera']['roll_degrees'] = first['roll_degrees']
                validate(bundle)
                wire = compact_response(bundle)
                decoded = expand_response(wire)
                validate(decoded)
                self.assertEqual(decoded, canonical_ignored(bundle))
                self.assertLess(len(json.dumps(wire)), len(json.dumps(bundle)))
                self.assertEqual(decoded['views'], bundle['views'])
                self.assertEqual(decoded['texture_sources'], bundle['texture_sources'])

    def test_sphere_ellipsoid_and_ignored_size_axes_preserved(self):
        original = bundle_with_parts(all_types())
        decoded = expand_response(compact_response(original))
        before = original['shared_scene']['custom_objects'][0]['parts']
        after = decoded['shared_scene']['custom_objects'][0]['parts']
        for index in (1, 2, 3, 6):
            self.assertEqual(before[index]['size'], after[index]['size'])
        self.assertEqual(after[2]['radius'], .1)
        self.assertEqual(after[2]['size'], [.3, .5, .7])
        # Partial positive sphere dimensions use radius; preserve even these
        # values rather than guessing an ellipsoid or dropping supplied size.
        odd = bundle_with_parts([part('sphere', size=[.2, 0, .4], radius=.1)])
        self.assertEqual(expand_response(compact_response(odd)), odd)

    def test_support_aliases_are_retained_for_existing_normalization(self):
        bundle = bundle_with_parts([part('box', size=[.1, .1, .1])])
        base = bundle['shared_scene']['base_scene']['objects'][0]
        base['repeat'] = [2, 1]
        base['spacing'] = [1, 0]
        bundle['shared_scene']['custom_objects'][0]['support_id'] = base['id']
        decoded = expand_response(compact_response(bundle))
        self.assertEqual(decoded, bundle)
        # The codec deliberately does not choose one of the repeated supports.
        # This remains the existing normalizer's operation with provenance.
        self.assertEqual(decoded['shared_scene']['custom_objects'][0]['support_id'], base['id'])

    def test_explicitly_ignored_nondefault_fields_are_canonicalized(self):
        values = [part('box', size=[1, 1, 1], radius=.4),
                  part('curve', size=[2, 3, 4], points=[[0, 0, 0], [1, 0, 0]], radius=.02),
                  part('mesh', size=[2, 3, 4], radius=.4,
                       vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]),
                  part('image_plane', size=[1, .1, 2], radius=.4, material_id='photo')]
        original = bundle_with_parts(values)
        decoded = expand_response(compact_response(original))
        actual = decoded['shared_scene']['custom_objects'][0]['parts']
        self.assertEqual(actual[0]['radius'], 0)
        self.assertEqual(actual[1]['size'], [0, 0, 0])
        self.assertEqual(actual[2]['size'], [0, 0, 0])
        self.assertEqual(actual[2]['radius'], 0)
        self.assertEqual(actual[2]['vertices'], values[2]['vertices'])
        self.assertEqual(actual[2]['faces'], values[2]['faces'])
        self.assertEqual(actual[3]['size'], values[3]['size'])
        self.assertEqual(actual[3]['radius'], 0)
        from staged_analysis import _detail_part_world_bounds
        def detail(item):
            return {'type': item['type'],
                    'data': item['position'] + item['size'] + item['rotation_degrees'] + [item['radius']],
                    'points': item['points'], 'vertices': item['vertices'],
                    'faces': item['faces'], 'cloth': item['cloth']}
        group = {'p': [.2, .5, .3], 'r': [13, 27, -8]}
        for before, after in zip(values, actual):
            self.assertEqual(_detail_part_world_bounds(detail(before), group),
                             _detail_part_world_bounds(detail(after), group))

    def test_schema_strict_branches_and_identity_restrictions_are_retained(self):
        views = [{'id': 'x', 'source_image': '/trusted/photo.jpg', 'width': 1234, 'height': 567}]
        full = response_schema(views, 'immutable_scene')
        before = copy.deepcopy(full)
        compact = compact_response_schema(full)
        self.assertEqual(full, before)
        self.assertEqual(compact['properties']['views'], full['properties']['views'])
        self.assertEqual(compact['properties']['scene_id'], full['properties']['scene_id'])
        self.assertEqual(compact['properties']['shared_scene']['properties']['image'],
                         full['properties']['shared_scene']['properties']['image'])
        branches = compact['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']['items']['anyOf']
        self.assertEqual(len(branches), 6)
        self.assertEqual([branch['properties']['type']['enum'] for branch in branches],
                         [['box', 'image_plane'], ['sphere'], ['sphere', 'cylinder'],
                          ['curve', 'polyline'], ['mesh'], ['curtain']])
        for branch in branches:
            kind = branch['properties']['type']['enum'][0]
            self.assertIn(set(branch['properties']), [set(COMMON + fields) for fields in FIELD_VARIANTS[kind]])
            self.assertEqual(set(branch['required']), set(branch['properties']))
            self.assertIs(branch['additionalProperties'], False)
            if kind == 'mesh':
                self.assertEqual(branch['properties']['vertices']['minItems'], 3)
                self.assertEqual(branch['properties']['faces']['minItems'], 1)
            if kind in ('curve', 'polyline'):
                self.assertEqual(branch['properties']['points']['minItems'], 2)
            if kind == 'curtain':
                self.assertEqual(branch['properties']['cloth']['minItems'], 1)
        compact['properties']['views']['minItems'] = 10
        self.assertEqual(full, before)

    def test_equivalent_type_branch_merge_preserves_wire_acceptance(self):
        bundle = bundle_with_parts(all_types())
        bundle['shared_scene'].setdefault('base_material_overrides', [])
        bundle['shared_scene'].setdefault('exterior_contexts', [])
        first = bundle['views'][0]
        bundle['shared_scene']['image'] = {
            'width': first['width'], 'height': first['height'], 'source_id': first['id']}
        views = [{'id': view['id'], 'source_image': 'input:' + view['id'],
                  'width': view['width'], 'height': view['height']}
                 for view in bundle['views']]
        compact = compact_response_schema(response_schema(views, bundle['scene_id']))
        parts_rule = compact['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']['items']
        merged_branches = parts_rule['anyOf']
        # Recreate the pre-dedup schema by splitting only the merged type enums.
        # Every other property and constraint remains byte-for-byte identical.
        legacy = copy.deepcopy(compact)
        legacy_parts = legacy['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']['items']
        legacy_parts['anyOf'] = []
        for branch in merged_branches:
            enum = branch['properties']['type']['enum']
            # curve/polyline was already one branch before this optimization;
            # the new merge only combines branches that were separate.
            kinds = enum if enum != ['curve', 'polyline'] else [enum]
            for kind in kinds:
                split = copy.deepcopy(branch)
                split['properties']['type']['enum'] = kind if isinstance(kind, list) else [kind]
                legacy_parts['anyOf'].append(split)
        self.assertEqual(len(legacy_parts['anyOf']), 8)
        for view in bundle['views']:
            view['source_image'] = 'input:' + view['id']
        first = bundle['views'][0]
        bundle['shared_scene']['image'] = {
            'width': first['width'], 'height': first['height'], 'source_id': first['id']}
        wire = compact_response(bundle)
        model_api._walk_wire(wire, compact)
        model_api._walk_wire(wire, legacy)
        mutations = [
            ('unknown_type', lambda item: item.update(type='pyramid')),
            ('wrong_geometry', lambda item: item.update(radius=1)),
            ('missing_common', lambda item: item.pop('material_id')),
            ('extra_field', lambda item: item.update(untrusted=1)),
            ('bad_mesh', lambda item: item.update(faces=[])),
        ]
        for name, mutate in mutations:
            candidate = copy.deepcopy(wire)
            mutate(candidate['shared_scene']['custom_objects'][0]['parts'][0 if name != 'bad_mesh' else 5])
            outcomes = []
            for schema in (compact, legacy):
                try:
                    model_api._walk_wire(candidate, schema)
                except ValueError:
                    outcomes.append(False)
                else:
                    outcomes.append(True)
            with self.subTest(mutation=name):
                self.assertEqual(outcomes, [False, False])

    def test_schema_drift_and_double_encoding_are_rejected(self):
        for mutation in ('extra_field', 'unknown_type', 'missing_field', 'not_strict', 'v4'):
            full = copy.deepcopy(SCHEMA)
            part_rule = full['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']['items']
            if mutation == 'extra_field':
                part_rule['properties']['new_geometry'] = {'type': 'number'}
                part_rule['required'].append('new_geometry')
            elif mutation == 'unknown_type':
                part_rule['properties']['type']['enum'].append('pyramid')
            elif mutation == 'missing_field':
                del part_rule['properties']['size']
            elif mutation == 'not_strict':
                part_rule['additionalProperties'] = True
            else:
                full['properties']['schema_version']['enum'] = [4]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compact_response_schema(full)
        with self.assertRaises(ValueError):
            compact_response_schema(compact_response_schema(SCHEMA))
        with self.assertRaises(ValueError):
            compact_response(compact_response(fixture()))

    def test_unknown_fields_missing_fields_and_inapplicable_fields_rejected(self):
        wire = compact_response(bundle_with_parts(all_types()))
        for change in ('unknown_top', 'unknown_view', 'unknown_group', 'extra_part', 'unused_array', 'missing_size'):
            bad = copy.deepcopy(wire)
            if change == 'unknown_top':
                bad['command'] = 'untrusted'
            elif change == 'unknown_view':
                bad['views'][0]['unexpected'] = 1
            elif change == 'unknown_group':
                bad['shared_scene']['custom_objects'][0]['unexpected'] = 1
            else:
                item = bad['shared_scene']['custom_objects'][0]['parts'][0]
                if change == 'extra_part': item['unexpected'] = 1
                elif change == 'unused_array': item['faces'] = []
                else: del item['size']
            with self.subTest(change=change), self.assertRaises(ValueError):
                expand_response(bad)
        for marker in (None, '', 'typed_parts_v2', 1):
            bad = copy.deepcopy(wire)
            if marker is None: del bad[FORMAT_FIELD]
            else: bad[FORMAT_FIELD] = marker
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                expand_response(bad)
        with self.assertRaises(ValueError):
            expand_response(bundle_with_parts(all_types()))

    def test_malformed_arrays_numbers_topology_never_filled_or_fixed(self):
        cases = [
            ('mesh', 'faces', []), ('mesh', 'vertices', [[0, 0, 0], [1, 0, 0]]),
            ('mesh', 'faces', [[0, 1, 5]]), ('mesh', 'faces', [[0, 1, 1]]),
            ('mesh', 'faces', [[0, 1, True]]), ('mesh', 'faces', [[0, 1, 2.0]]),
            ('mesh', 'vertices', [[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            ('curve', 'points', [[0, 0, 0]]), ('curve', 'points', [[0, 0, 0], [0, 0, 0]]),
            ('curve', 'radius', 0), ('curve', 'radius', float('nan')),
            ('polyline', 'points', [[0, 0, 0]]), ('polyline', 'points', [[0, 0, 0], [0, 0, 0]]),
            ('polyline', 'radius', 0), ('polyline', 'radius', float('nan')),
            ('box', 'size', [1, 1]), ('box', 'size', [1, 0, 1]),
            ('box', 'type', 'unknown'), ('box', 'type', ['box']),
            ('sphere', 'size', 'bad'), ('sphere', 'radius', True),
            ('curtain', 'cloth', []), ('curtain', 'cloth', {}),
            ('curtain', 'size', [1, .01, 1]),
        ]
        wire = compact_response(bundle_with_parts(all_types()))
        for kind, field, value in cases:
            bad = copy.deepcopy(wire)
            selected = next(item for item in bad['shared_scene']['custom_objects'][0]['parts'] if item['type'] == kind)
            selected[field] = value
            before = copy.deepcopy(bad)
            with self.subTest(kind=kind, field=field, value=value), self.assertRaises(ValueError):
                expand_response(bad)
            self.assertEqual(json.dumps(bad), json.dumps(before))
        for field, value in (('fold_depth', .4), ('segments_x', 8), ('sag', 1)):
            bad = copy.deepcopy(wire)
            curtain = next(item for item in bad['shared_scene']['custom_objects'][0]['parts']
                           if item['type'] == 'curtain')
            curtain['cloth'][0][field] = value
            with self.subTest(cloth=field), self.assertRaises(ValueError):
                expand_response(bad)

    def test_encoder_rejects_invalid_geometry_and_unknown_fields(self):
        for value in (part('box', size=[1, 1, 1], points=[[0, 0, 0]]),
                      part('box', size=[1, 1, 1], vertices=[[0, 0, 0]]),
                      part('box', size=[1, 1, 1], faces=[[0, 1, 2]]),
                      part('box', size=[1, 1, 1], cloth=part('curtain')['cloth']),
                      part('mesh', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                      part('curtain', size=[1, .2, 1], radius=.1),
                      {**part('box', size=[1, 1, 1]), 'code': 'untrusted'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compact_response(bundle_with_parts([value]))

    def test_reference_material_and_geometry_budget_validations_remain(self):
        wire = compact_response(bundle_with_parts(all_types()))
        bad = copy.deepcopy(wire)
        bad['shared_scene']['custom_objects'][0]['parts'][0]['material_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'reference'):
            expand_response(bad)
        bad = copy.deepcopy(wire)
        bad['shared_scene']['custom_objects'][0]['parts'][6]['material_id'] = 'wood'
        with self.assertRaisesRegex(ValueError, 'photo texture'):
            expand_response(bad)
        bad = bundle_with_parts([part('box', size=[1, 1, 1])])
        template = bad['shared_scene']['custom_objects'][0]
        bad['shared_scene']['custom_objects'] = [
            {**copy.deepcopy(template), 'id': 'g_' + str(i), 'parts': [part('box', size=[1, 1, 1])] * 100}
            for i in range(9)]
        with self.assertRaisesRegex(ValueError, 'budget'):
            compact_response(bad)

    def test_view_identity_tampering_reaches_existing_normalizer(self):
        bundle = bundle_with_parts(all_types())
        views = [{'id': view['id'], 'source_image': '/trusted/' + view['id'] + '.png',
                  'width': view['width'], 'height': view['height']} for view in bundle['views']]
        for view in bundle['views']:
            view['source_image'] = 'input:' + view['id']
        first = views[0]
        bundle['shared_scene']['image'] = {'width': first['width'], 'height': first['height'], 'source_id': first['id']}
        wire = compact_response(bundle)
        for key, value in (('id', 'changed'), ('source_image', 'input:other'), ('width', 500)):
            bad = copy.deepcopy(wire)
            bad['views'][0][key] = value
            expanded = expand_response(bad)
            self.assertEqual(expanded['views'][0][key], value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Model changed'):
                normalize_response(expanded, views, bundle['scene_id'])
        expanded = expand_response(wire)
        normalized, _ = normalize_response(expanded, views, bundle['scene_id'])
        self.assertEqual(normalized['views'][0]['source_image'], views[0]['source_image'])
        self.assertEqual(expanded, bundle)


if __name__ == '__main__':
    unittest.main()
