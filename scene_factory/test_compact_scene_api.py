"""Real loopback HTTP coverage for compact scene wire transport; no paid API."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from compact_scene import FORMAT_FIELD, FORMAT_VERSION, compact_response, compact_response_schema, expand_response
from factory_config import normalize_config
import model_api as api
from multiview_spec import validate
from test_compact_scene import all_types, bundle_with_parts
from test_model_api import FixtureServer, Handler, PNG, chat
import vision_multiview as vision


def complete_bundle():
    """The provider's strict wire requires additive V2 fields, unlike storage."""
    bundle = bundle_with_parts(all_types())
    scene = bundle['shared_scene']
    scene.setdefault('base_material_overrides', [])
    scene.setdefault('exterior_contexts', [])
    first = bundle['views'][0]
    scene['image'] = {'width': first['width'], 'height': first['height'], 'source_id': first['id']}
    scene['base_scene']['camera'] = {key: first['camera'][key]
                                     for key in ('position', 'target', 'lens_mm')}
    scene['camera']['roll_degrees'] = first['camera']['roll_degrees']
    for view in bundle['views']:
        view['source_image'] = 'input:' + view['id']
    return bundle


def parts_schema(schema):
    return schema['properties']['shared_scene']['properties']['custom_objects']['items']['properties']['parts']


class CompactSceneAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FixtureServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.image = self.directory / 'reference.png'
        self.image.write_bytes(PNG)
        self.config = {'provider': 'openai_chat', 'model': 'compact-fixture',
                       'base_url': f'http://127.0.0.1:{self.server.server_port}/v1',
                       'api_key_env': '', 'max_retries': 0}
        self.server.requests, self.server.responses = [], []
        self.bundle = complete_bundle()
        self.wire = compact_response(self.bundle)
        self.schema = compact_response_schema(vision.SCHEMA)
        self.sequence = 0

    def request(self, wire=None, schema=None, config=None, response=None):
        wire = self.wire if wire is None else wire
        self.server.responses.append((200, {}, chat(json.dumps(wire)) if response is None else response))
        self.sequence += 1
        self.output = self.directory / f'request_{self.sequence}.json'
        return api.call_model('Return one complete scene with typed parts.', [self.image],
                              self.schema if schema is None else schema, self.output,
                              config=self.config if config is None else config)

    def record(self):
        return json.loads(self.output.with_suffix('.api.json').read_text())

    def test_all_nine_part_values_pass_real_transport_and_semantic_decode(self):
        # Includes sphere radius-only, ellipsoid size+radius, smooth curves,
        # rigid polylines, and all other types via eight strict wire branches.
        result = self.request()
        self.assertEqual(result, self.wire)
        self.assertEqual(expand_response(result), self.bundle)
        validate(expand_response(result))
        self.assertEqual(len(result['shared_scene']['custom_objects'][0]['parts']), 9)
        self.assertEqual(json.loads(self.output.with_suffix('.raw.json').read_text()), self.wire)
        self.assertEqual(self.record()['status'], 'completed')
        self.assertEqual(self.record()['usage']['total_tokens'], 19)
        self.assertEqual(len(self.server.requests), 1)
        payload = self.server.requests[0]['body']
        self.assertEqual(payload['response_format']['json_schema']['schema'], self.schema)
        self.assertIs(payload['response_format']['json_schema']['strict'], True)
        self.assertEqual(len(parts_schema(self.schema)['items']['anyOf']), 6)

    def test_supported_provider_modes_carry_same_strict_union(self):
        for provider, mode in (('openai_chat', 'json_object'), ('openai_chat', 'prompt'),
                               ('openai_responses', 'json_schema'), ('openai_responses', 'prompt'),
                               ('anthropic', 'prompt')):
            config = {**self.config, 'provider': provider, 'response_format': mode}
            if provider == 'openai_responses':
                response = {'status': 'completed', 'output': [
                    {'type': 'message', 'status': 'completed', 'content': [
                        {'type': 'output_text', 'text': json.dumps(self.wire)}]}]}
            elif provider == 'anthropic':
                response = {'type': 'message', 'stop_reason': 'end_turn', 'content': [
                    {'type': 'text', 'text': json.dumps(self.wire)}]}
            else:
                response = chat(json.dumps(self.wire))
            with self.subTest(provider=provider, mode=mode):
                self.assertEqual(expand_response(self.request(config=config, response=response)), self.bundle)
                payload = self.server.requests[-1]['body']
                if mode == 'json_schema':
                    self.assertEqual(payload['text']['format']['schema'], self.schema)
                else:
                    content = (payload['input'] if provider == 'openai_responses' else payload['messages'])[0]['content']
                    prompt = content[0]['text']
                    self.assertIn('"anyOf"', prompt)
                    self.assertIn('"typed_parts_v1"', prompt)

    def test_missing_faces_empty_geometry_and_cross_type_fields_rejected(self):
        for mutation in ('missing_faces', 'empty_faces', 'two_vertices', 'one_curve_point', 'one_polyline_point',
                         'box_faces', 'curve_vertices', 'sphere_cloth', 'mesh_size', 'missing_cloth',
                         'missing_common', 'unknown_field', 'duplicate_json_key'):
            wire = copy.deepcopy(self.wire)
            parts = wire['shared_scene']['custom_objects'][0]['parts']
            if mutation == 'missing_faces': del parts[5]['faces']
            elif mutation == 'empty_faces': parts[5]['faces'] = []
            elif mutation == 'two_vertices': parts[5]['vertices'] = [[0, 0, 0], [1, 0, 0]]
            elif mutation == 'one_curve_point': parts[4]['points'] = [[0, 0, 0]]
            elif mutation == 'one_polyline_point': parts[8]['points'] = [[0, 0, 0]]
            elif mutation == 'box_faces': parts[0]['faces'] = []
            elif mutation == 'curve_vertices': parts[4]['vertices'] = []
            elif mutation == 'sphere_cloth': parts[1]['cloth'] = []
            elif mutation == 'mesh_size': parts[5]['size'] = [0, 0, 0]
            elif mutation == 'missing_cloth': del parts[7]['cloth']
            elif mutation == 'missing_common': del parts[0]['material_id']
            elif mutation == 'unknown_field': parts[0]['code'] = 'do not execute'
            response = None
            if mutation == 'duplicate_json_key':
                text = json.dumps(wire).replace('"wire_format": "typed_parts_v1"',
                    '"wire_format": "typed_parts_v1", "wire_format": "typed_parts_v1"')
                response = chat(text)
            before = len(self.server.requests)
            with self.subTest(mutation=mutation), self.assertRaises(api.ModelAPIError) as caught:
                self.request(wire, response=response)
            self.assertEqual(caught.exception.code, 'invalid_json' if mutation == 'duplicate_json_key' else 'schema_mismatch')
            self.assertEqual(len(self.server.requests), before + 1)
            self.assertEqual(self.record()['status'], 'failed')
            self.assertFalse(self.output.with_suffix('.raw.json').exists())

    def test_union_enforces_enum_types_bounds_and_strings(self):
        mutations = {
            'bad_type_enum': lambda b, p: p[0].update(type='pyramid'),
            'bad_marker_enum': lambda b, p: b.update(wire_format='typed_parts_v2'),
            'bool_as_radius': lambda b, p: p[1].update(radius=True),
            'string_as_vector': lambda b, p: p[0].update(position='0,0,0'),
            'negative_size': lambda b, p: p[0].update(size=[1, -1, 1]),
            'size_max': lambda b, p: p[0].update(size=[31, 1, 1]),
            'radius_max': lambda b, p: p[1].update(radius=6),
            'rotation_max': lambda b, p: p[0].update(rotation_degrees=[0, 0, 361]),
            'long_material': lambda b, p: p[0].update(material_id='a' * 129),
            'bool_face_index': lambda b, p: p[5].update(faces=[[0, 1, True]]),
            'fractional_face_index': lambda b, p: p[5].update(faces=[[0, 1, 2.0]]),
            'float_vertex_overflow': lambda b, p: p[5].update(vertices=[[0, 0, 0], [51, 0, 0], [0, 0, 1]]),
            'cloth_minimum': lambda b, p: p[7]['cloth'][0].update(fold_count=1),
            'cloth_unknown': lambda b, p: p[7]['cloth'][0].update(extra=1),
            'empty_scene_id': lambda b, p: b.update(scene_id=''),
            'scene_id_pattern': lambda b, p: b.update(scene_id='../../scene'),
            'view_id_pattern': lambda b, p: b['views'][0].update(id='space name'),
            'camera_focal_bounds': lambda b, p: b['views'][0]['camera'].update(lens_mm=501),
            'bool_not_integer': lambda b, p: b['views'][0].update(width=True),
            'missing_v2_extension': lambda b, p: b['shared_scene'].pop('exterior_contexts'),
        }
        for name, mutate in mutations.items():
            wire = copy.deepcopy(self.wire)
            mutate(wire, wire['shared_scene']['custom_objects'][0]['parts'])
            with self.subTest(case=name), self.assertRaises(api.ModelAPIError) as caught:
                self.request(wire)
            self.assertEqual(caught.exception.code, 'schema_mismatch')
            self.assertFalse(self.output.with_suffix('.raw.json').exists())

    def test_transport_does_not_replace_semantic_geometry_validation(self):
        for mutation in ('invalid_index', 'repeated_index', 'degenerate_face', 'coincident_curve',
                         'zero_curve_radius', 'coincident_polyline', 'zero_polyline_radius',
                         'cloth_envelope', 'cloth_resolution', 'unknown_material'):
            wire = copy.deepcopy(self.wire)
            parts = wire['shared_scene']['custom_objects'][0]['parts']
            if mutation == 'invalid_index': parts[5]['faces'] = [[0, 1, 3]]
            elif mutation == 'repeated_index': parts[5]['faces'] = [[0, 1, 1]]
            elif mutation == 'degenerate_face': parts[5]['vertices'] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
            elif mutation == 'coincident_curve': parts[4]['points'] = [[0, 0, 0], [0, 0, 0]]
            elif mutation == 'zero_curve_radius': parts[4]['radius'] = 0
            elif mutation == 'coincident_polyline': parts[8]['points'] = [[0, 0, 0], [0, 0, 0]]
            elif mutation == 'zero_polyline_radius': parts[8]['radius'] = 0
            elif mutation == 'cloth_envelope': parts[7]['cloth'][0]['fold_depth'] = .5
            elif mutation == 'cloth_resolution': parts[7]['cloth'][0]['segments_x'] = 8
            elif mutation == 'unknown_material': parts[0]['material_id'] = 'not_defined'
            returned = self.request(wire)
            self.assertEqual(returned, wire)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                expand_response(returned)
            # A valid wire response is retained as evidence even if a later
            # geometry check fails; no second model request or topology repair.
            self.assertEqual(self.record()['status'], 'completed')
            self.assertEqual(json.loads(self.output.with_suffix('.raw.json').read_text()), wire)
            self.assertFalse(self.output.exists())

    def test_malformed_unions_fail_before_any_http_request(self):
        branch = copy.deepcopy(parts_schema(self.schema)['items']['anyOf'][0])
        bad_unions = [
            {'anyOf': []}, {'anyOf': [branch]}, {'anyOf': [branch] * 11},
            {'anyOf': 'not_array'}, {'anyOf': [branch, {'type': 'string'}]},
            {'anyOf': [branch, {}]}, {'anyOf': [branch, None]},
            {'anyOf': [branch, branch], 'type': 'object'},
            {'anyOf': [branch, branch], 'description': 'unsupported union sibling'},
            {'anyOf': [branch, {**branch, 'additionalProperties': True}]},
            {'anyOf': [branch, {**branch, 'required': ['type']}]},
            {'anyOf': [branch, {**branch, 'unknownKeyword': True}]},
        ]
        for union in bad_unions:
            schema = copy.deepcopy(self.schema)
            parts_schema(schema)['items'] = union
            with self.subTest(union=union), self.assertRaises(ValueError):
                api.call_model('Test invalid schema.', [self.image], schema, self.directory / 'invalid.json', config=self.config)
        with self.assertRaises(ValueError):
            api.call_model('Test invalid root union.', [self.image], {'anyOf': [branch, branch]},
                           self.directory / 'invalid_root.json', config=self.config)
        self.assertEqual(self.server.requests, [])
        self.assertFalse((self.directory / 'invalid.raw.json').exists())

    def test_pipeline_config_enables_compact_analysis_through_real_api(self):
        config = normalize_config({'schema_version': 1, 'models': {'analysis': self.config},
            'pipeline': {'analysis_mode': 'monolithic', 'scene_wire_format': FORMAT_VERSION,
                         'validation_retries': 0}, 'batch': {'no_review': True, 'max_repairs': 0}})
        self.assertEqual(config['pipeline']['scene_wire_format'], FORMAT_VERSION)
        self.assertTrue(config['batch']['no_review'])
        views = [{**{k: v[k] for k in ('id', 'width', 'height')}, 'source_image': str(self.image)}
                 for v in self.bundle['views']]
        self.server.responses.append((200, {}, chat(json.dumps(self.wire))))
        output = self.directory / 'initial_bundle.json'
        result = vision.analyze('room', views, output, model_config=config['models']['analysis'],
                                wire_format=config['pipeline']['scene_wire_format'])
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(result['shared_scene'], self.bundle['shared_scene'])
        self.assertNotIn(FORMAT_FIELD, result)
        validate(result)
        self.assertEqual(json.loads(output.with_suffix('.raw.json').read_text()), self.wire)
        provenance = json.loads((self.directory / 'analysis_provenance.json').read_text())
        self.assertEqual(provenance['scene_wire_encoding']['format'], FORMAT_VERSION)
        self.assertFalse(provenance['manual_scene_seed'])
        defaults = normalize_config({'schema_version': 1, 'models': {'analysis': self.config}})
        self.assertEqual(defaults['pipeline']['scene_wire_format'], 'canonical')
        for pipeline in ({'scene_wire_format': 'unknown'},
                         {'scene_wire_format': FORMAT_VERSION, 'analysis_mode': 'staged'}):
            with self.subTest(pipeline=pipeline), self.assertRaises(ValueError):
                normalize_config({'schema_version': 1, 'models': {'analysis': self.config}, 'pipeline': pipeline})


if __name__ == '__main__':
    unittest.main()
