"""Bounded asset-selection evidence for visual assessment and typed repair."""
from pathlib import Path

from batch import file_sha256, read_json


ASSET_GUIDANCE = '''External library bindings override the base object's procedural geometry and color. The fitted dimensions in asset_selection are the actual library-size estimate. Moving/rotating a base object changes its placement; changing its target dimensions triggers fresh selection on the next iteration. Editing its procedural color does not recolor a bound library material. You cannot edit a library mesh, invent asset IDs or change binding paths through scene edits. If the selected shape or material cannot match the photo with available scene edits, report the specific visible asset limitation. A procedural fallback is a library coverage gap, not by itself a visual failure; judge the actual paired images. No asset selection is a certificate of contact, collision or appearance correctness.'''


def snapshot_inputs(directory, *, include_metadata=False):
    """Freeze optional evidence, including absence so new files cannot go unseen."""
    directory = Path(directory)
    names = ('asset_bindings.json', 'asset_selection.json')
    if include_metadata:
        names += ('metadata.json',)
    return {name: file_sha256(directory / name) if (directory / name).is_file() else None
            for name in names}


def load_context(directory):
    """Return a small model payload and hashes; omit candidate pools and paths."""
    directory = Path(directory)
    names = ('asset_bindings.json', 'asset_selection.json')
    present = [(directory / name).is_file() for name in names]
    if not any(present):
        return {}, {}
    if not all(present):
        raise ValueError('Automatic asset context requires both bindings and selection evidence')
    hashes = {name: file_sha256(directory / name) for name in names}
    bindings, report = (read_json(directory / name) for name in names)
    context = {'instances': bindings['instances'], 'appearance_match_verified': False,
               'objects': {identifier: {key: record[key] for key in
                   ('status', 'category', 'target_dimensions_m', 'fit', 'reasons', 'support', 'preview_only')
                   if key in record} for identifier, record in report['objects'].items()},
               'coverage': {key: report[key] for key in
                   ('total_objects', 'bound_objects', 'fallback_objects', 'preview') if key in report}}
    return context, hashes
