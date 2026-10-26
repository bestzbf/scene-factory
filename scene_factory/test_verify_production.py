"""Real loopback API tests for the retained batch smoke verification."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scene_factory.verify_production import verify
from batch import file_sha256, read_json


class ProductionVerificationTests(unittest.TestCase):
    def test_api_review_repair_final_gate_and_resume(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / 'new-fixture'
            result = verify(output, mock_render=True)
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['synthetic_fixture'])
            self.assertFalse(result['real_blender'])
            self.assertFalse(result['photograph_reconstruction_validated'])
            self.assertEqual((result['api_requests'], result['blender_build_invocations']), (11, 4))
            self.assertEqual((result['api_requests_on_resume'], result['blender_build_invocations_on_resume']), (0, 0))
            self.assertEqual(result['rendered_views'], 6)
            self.assertEqual(result['accepted_fixture_scene_ids'], ['single_accepted', 'two_views'])
            self.assertTrue(result['separately_configured_api_models_verified'])
            self.assertTrue(result['ordered_image_hashes_verified'])
            requests = result['request_evidence']
            self.assertEqual(sum(row['stage'] == 'repair' for row in requests), 1)
            self.assertTrue(all(row['image_count'] == row['view_count'] * (2 if row['stage'] in ('review', 'repair') else 1)
                                for row in requests))
            for row in requests:
                if row['stage'] not in ('review', 'repair'):
                    continue
                self.assertEqual(len(row['model_image_inputs']), row['view_count'])
                for index, image_input in enumerate(row['model_image_inputs']):
                    render = Path(image_input['render_path'])
                    model_input = Path(image_input['model_input_path'])
                    self.assertEqual(image_input['render_sha256'], file_sha256(render))
                    self.assertEqual(image_input['model_input_sha256'], file_sha256(model_input))
                    self.assertEqual(row['ordered_image_sha256'][2 * index + 1], image_input['model_input_sha256'])
                    provenance_name = 'review_provenance.json' if row['stage'] == 'review' else 'repair_provenance.json'
                    provenance = read_json(render.parent / provenance_name)
                    bound_input = next(item for item in provenance['model_image_inputs']
                                       if item['view_id'] == image_input['view_id'])
                    for key in ('view_id', 'model_input_path', 'model_input_sha256',
                                'render_sha256'):
                        self.assertEqual(bound_input[key], image_input[key])
                    if row['stage'] == 'review':
                        self.assertEqual(provenance['input_hashes'][render.name], image_input['render_sha256'])
                    else:
                        repair_input = next(item for item in provenance['image_sha256']
                                            if item['view_id'] == image_input['view_id'])
                        self.assertEqual(repair_input['render_sha256'], image_input['render_sha256'])
            self.assertEqual(read_json(output / 'validation.json'), result)
            with self.assertRaises(FileExistsError):
                verify(output, mock_render=True)


if __name__ == '__main__':
    unittest.main()
