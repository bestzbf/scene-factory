"""Deterministic source-preserving render proxies for model image inputs."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from model_image_inputs import IMAGE_INPUT_POLICY, prepare_render_input


class ModelImageInputTests(unittest.TestCase):
    def _image(self, path, size, color=(20, 40, 60)):
        Image.new('RGB', size, color).save(path)
        return path

    def test_downsamples_only_matching_oversized_render_and_binds_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration = root/'iteration'
            iteration.mkdir()
            source_render = self._image(iteration/'rgb_a.png', (2560, 1920))
            original_bytes = source_render.read_bytes()
            view = {'id': 'a', 'width': 1280, 'height': 960}

            model_input, record = prepare_render_input(view, source_render, iteration)

            self.assertEqual(model_input, (iteration/'model_input_images/rgb_a.png').resolve())
            with Image.open(model_input) as image:
                self.assertEqual(image.size, (1280, 960))
            self.assertEqual(source_render.read_bytes(), original_bytes)
            self.assertEqual(record['policy'], IMAGE_INPUT_POLICY)
            self.assertEqual(record['render_sha256'], hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual(record['model_input_sha256'], hashlib.sha256(model_input.read_bytes()).hexdigest())
            self.assertEqual(record['transform'], 'render_only_lanczos_downsample_to_reference_dimensions')
            self.assertEqual(prepare_render_input(view, source_render, iteration), (model_input, record))

    def test_never_upscales_crops_or_distorts_render(self):
        view = {'id': 'a', 'width': 1280, 'height': 960}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration = root/'iteration'
            iteration.mkdir()
            for size in ((640, 480), (1000, 750), (2560, 1080), (1280, 960)):
                with self.subTest(size=size):
                    render = self._image(iteration/'render.png', size)
                    returned, record = prepare_render_input(view, render, iteration)
                    self.assertEqual(returned, render.resolve())
                    self.assertEqual(record['transform'], 'none')
                    self.assertEqual(record['model_input_dimensions_px'], list(size))
            large_matching = self._image(iteration/'large.png', (1600, 1200))
            returned, record = prepare_render_input(view, large_matching, iteration)
            self.assertNotEqual(returned, large_matching.resolve())
            self.assertEqual(record['model_input_dimensions_px'], [1280, 960])
            self.assertEqual(record['transform'], 'render_only_lanczos_downsample_to_reference_dimensions')

    def test_changed_render_cannot_reuse_stale_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration = root/'iteration'
            iteration.mkdir()
            render = self._image(iteration/'rgb_a.png', (2560, 1920), (20, 40, 60))
            view = {'id': 'a', 'width': 1280, 'height': 960}
            prepare_render_input(view, render, iteration)
            render.unlink()
            self._image(render, (2560, 1920), (100, 120, 140))
            with self.assertRaisesRegex(ValueError, 'differs from current source'):
                prepare_render_input(view, render, iteration)

    def test_proxy_symlink_cannot_escape_iteration_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration, external = root/'iteration', root/'external'
            iteration.mkdir()
            external.mkdir()
            render = self._image(iteration/'rgb_a.png', (2560, 1920))
            (iteration/'model_input_images').symlink_to(external, target_is_directory=True)
            view = {'id': 'a', 'width': 1280, 'height': 960}
            with self.assertRaisesRegex(ValueError, 'symbolic link'):
                prepare_render_input(view, render, iteration)

    def test_render_symlink_cannot_upload_external_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration, external = root/'iteration', root/'external'
            iteration.mkdir()
            external.mkdir()
            self._image(external/'render.png', (1280, 960))
            (iteration/'rgb_a.png').symlink_to(external/'render.png')
            view = {'id': 'a', 'width': 1280, 'height': 960}
            with self.assertRaisesRegex(ValueError, 'Symbolic-link render input'):
                prepare_render_input(view, iteration/'rgb_a.png', iteration)


if __name__ == '__main__':
    unittest.main()
