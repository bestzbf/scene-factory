import copy
import unittest

from spec_v2 import validate
from test_spec_v2 import fixture


class ExteriorContextContractTests(unittest.TestCase):
    def fixture(self):
        spec = fixture()
        entry = spec['base_scene']['objects'][0]
        entry.update(category='window', repeat=[1, 1])
        spec['exterior_contexts'] = [{'window_id': entry['id'], 'texture_id': 'sign',
                                      'depth_m': .2, 'brightness_ev': 1}]
        return spec

    def test_valid_context_and_legacy_omission(self):
        spec = self.fixture(); validate(spec)
        spec.pop('exterior_contexts'); before = copy.deepcopy(spec)
        validate(spec); self.assertEqual(spec, before)

    def test_attachment_is_window_not_generic_object(self):
        spec = self.fixture(); spec['base_scene']['objects'][0]['category'] = 'desk'
        with self.assertRaisesRegex(ValueError, 'unknown reference'):
            validate(spec)

    def test_no_entire_photo_backdrop_or_missing_crop(self):
        spec = self.fixture(); spec['textures'][0]['quad'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        with self.assertRaisesRegex(ValueError, 'half the photo'):
            validate(spec)
        spec = self.fixture(); spec['exterior_contexts'][0]['texture_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unknown reference'):
            validate(spec)

    def test_duplicate_contexts_and_unbounded_parameters_rejected(self):
        spec = self.fixture(); spec['exterior_contexts'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate(spec)
        for key, value in [('depth_m', 0), ('depth_m', 21), ('brightness_ev', 7)]:
            spec = self.fixture(); spec['exterior_contexts'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(spec)


if __name__ == '__main__':
    unittest.main()
