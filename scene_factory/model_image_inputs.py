"""Deterministic API/CLI image inputs for matched scene reviews.

Reference photos remain untouched at their supplied pixel resolution. When a
Blender render is larger than the paired reference, create a review-only image
at reference resolution: extra render pixels cannot show source-supported
detail and otherwise make the model compare unequal-resolution pairs. The
authoritative render and acceptance metrics are never resized.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


IMAGE_INPUT_POLICY = 'matched_render_at_reference_resolution_v1'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_render_input(view: dict, render_path, directory) -> tuple[Path, dict]:
    """Return an immutable comparison image and exact resize provenance.

    Resize only when both render dimensions are at least the original view
    dimensions, at least one is larger, and aspect ratios agree within .1%.
    Thus this helper never upscales, crops, or distorts a rendered camera.
    """
    from PIL import Image

    original_render_path = Path(render_path)
    if original_render_path.is_symlink():
        raise ValueError('Symbolic-link render input is not allowed')
    render_path = original_render_path.resolve()
    directory = Path(directory).resolve()
    if not render_path.is_file() or not render_path.is_relative_to(directory):
        raise ValueError('Render model input must be an existing file inside its iteration directory')
    reference_size = (view['width'], view['height'])
    render_sha256 = _sha256(render_path)
    try:
        with Image.open(render_path) as source:
            source.load()
            render_size = source.size
            compatible_aspect = abs(
                render_size[0] / render_size[1]
                - reference_size[0] / reference_size[1]
            ) <= max(reference_size[0] / reference_size[1], 1.0) * .001
            resize = (
                render_size != reference_size
                and render_size[0] >= reference_size[0]
                and render_size[1] >= reference_size[1]
                and compatible_aspect
            )
            if not resize:
                return render_path, {
                    'view_id': view['id'],
                    'policy': IMAGE_INPUT_POLICY,
                    'render_sha256': render_sha256,
                    'render_dimensions_px': list(render_size),
                    'reference_dimensions_px': list(reference_size),
                    'model_input_path': str(render_path),
                    'model_input_sha256': render_sha256,
                    'model_input_dimensions_px': list(render_size),
                    'transform': 'none',
                }

            proxy_directory = directory / 'model_input_images'
            if proxy_directory.is_symlink():
                raise ValueError('Model-input render proxy directory may not be a symbolic link')
            output = proxy_directory / ('rgb_' + view['id'] + '.png')
            if output.is_symlink() or not output.resolve().is_relative_to(directory):
                raise ValueError('Model-input render proxy must remain inside its iteration directory')
            image = source.convert('RGB').resize(reference_size, Image.Resampling.LANCZOS)
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=False, compress_level=6)
            encoded = buffer.getvalue()
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith('Model-input render proxy'):
            raise
        raise ValueError('Render model input is not a valid image') from exc
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    if output.exists():
        if not output.is_file() or _sha256(output) != expected_sha256:
            raise ValueError('Existing model-input render proxy differs from current source; use a new iteration directory')
    else:
        _atomic_write(output, encoded)
    return output, {
        'view_id': view['id'],
        'policy': IMAGE_INPUT_POLICY,
        'render_sha256': render_sha256,
        'render_dimensions_px': list(render_size),
        'reference_dimensions_px': list(reference_size),
        'model_input_path': str(output),
        'model_input_sha256': expected_sha256,
        'model_input_dimensions_px': list(reference_size),
        'transform': 'render_only_lanczos_downsample_to_reference_dimensions',
    }
