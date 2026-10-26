"""Small, explicit model evidence derived from immutable local render reports.

These views are NOT validators or acceptance evidence. Callers must bind and
validate the complete source files. Nothing here edits the scene or the reports.
Large provenance/history/component dumps stay local; selector semantics, actual
fitted controls, diagnostic failures and image/view identities remain available.
"""
from __future__ import annotations

import copy
import json


CONTEXT_VERSION = 3
RECORD_TABLE_VERSION = 1
_RECORD_TABLE_MARKER = '$record_table_v1'

RECORD_TABLE_INSTRUCTION = (
    '\nScene-data encoding: some arrays of records may be represented as '
    '{"$record_table_v1":{"columns":[...],"rows":[...]}}. This is a lossless '
    'array encoding: each row supplies one record, its values map to columns by '
    'position, nested values keep their original meaning, and row order is unchanged. '
    'Interpret the decoded records exactly as ordinary JSON; do not treat columns as '
    'scene fields or infer omitted values. Other arrays and objects are unchanged. '
    'This encoding applies only to the input context. Any output governed by a JSON '
    'schema, especially revised_bundle, must use ordinary schema-compliant JSON arrays '
    'and objects; expand encoded records and never emit the $record_table_v1 wrapper.\n')


def encode_record_tables(value, *, sort_keys=False, exclude_keys=()):
    """Losslessly column-encode repetitive arrays of records when wire size falls.

    This changes only the model-facing JSON serialization. Callers retain and
    validate the original scene/report objects, so no identity, geometry or
    acceptance evidence is removed. Heterogeneous arrays and arrays whose
    encoded form is not smaller remain ordinary JSON arrays. The encoding is
    used only when aggregate savings exceed the UTF-8 size of its decoder note.
    """
    def encoded_json(item):
        return json.dumps(item, ensure_ascii=False, sort_keys=sort_keys,
                          separators=(',', ':'), allow_nan=False)

    original_bytes = len(encoded_json(value).encode('utf-8'))
    count = 0

    excluded = set(exclude_keys)

    def visit(item):
        nonlocal count
        if isinstance(item, dict):
            if _RECORD_TABLE_MARKER in item:
                raise ValueError('Reserved scene-data table marker is present in source data')
            return {key: (copy.deepcopy(child) if key in excluded else visit(child))
                    for key, child in item.items()}
        if not isinstance(item, list):
            return item
        rows = [visit(child) for child in item]
        if len(rows) < 2 or not all(isinstance(row, dict) for row in rows):
            return rows
        columns = sorted(rows[0])
        if any(sorted(row) != columns for row in rows):
            return rows
        candidate = {_RECORD_TABLE_MARKER: {
            'columns': columns,
            'rows': [[row[column] for column in columns] for row in rows],
        }}
        if len(encoded_json(candidate)) >= len(encoded_json(rows)):
            return rows
        count += 1
        return candidate

    encoded = visit(value)
    encoded_bytes = len(encoded_json(encoded).encode('utf-8'))
    instruction_bytes = len(RECORD_TABLE_INSTRUCTION.encode('utf-8')) if count else 0
    if count and original_bytes - encoded_bytes <= instruction_bytes:
        encoded = copy.deepcopy(value)
        encoded_bytes = original_bytes
        count = 0
        instruction_bytes = 0
    return encoded, {
        'version': RECORD_TABLE_VERSION,
        'encoded_arrays': count,
        'original_utf8_bytes': original_bytes,
        'encoded_utf8_bytes': encoded_bytes,
        'saved_utf8_bytes': original_bytes - encoded_bytes,
        'decoder_instruction_utf8_bytes': instruction_bytes,
        'net_saved_utf8_bytes_after_instruction': original_bytes - encoded_bytes - instruction_bytes,
    }


def decode_record_tables(value):
    """Decode the lossless model-facing record table form for independent checks."""
    if isinstance(value, dict):
        if _RECORD_TABLE_MARKER in value:
            if set(value) != {_RECORD_TABLE_MARKER}:
                raise ValueError('Malformed scene-data table wrapper')
            table = value[_RECORD_TABLE_MARKER]
            if (not isinstance(table, dict) or set(table) != {'columns', 'rows'}
                    or not isinstance(table['columns'], list)
                    or not all(isinstance(column, str) for column in table['columns'])
                    or len(table['columns']) != len(set(table['columns']))
                    or not isinstance(table['rows'], list)
                    or any(not isinstance(row, list) or len(row) != len(table['columns'])
                           for row in table['rows'])):
                raise ValueError('Malformed scene-data table columns or rows')
            return [{column: decode_record_tables(cell)
                     for column, cell in zip(table['columns'], row)}
                    for row in table['rows']]
        return {key: decode_record_tables(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_record_tables(child) for child in value]
    return value


def _pick(record, keys):
    if not isinstance(record, dict):
        return {}
    return {key: copy.deepcopy(record[key]) for key in keys if key in record}


def _identity(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def diagnostic_messages(value):
    """Retain every warnings/errors value, deduplicating identical repeated dumps.

    Sources locate diagnostics inside the report, not filesystem paths. No
    slicing or severity filtering is allowed here: unknown warning types remain
    visible. Missing, null and empty evidence are distinct.
    """
    found = {}

    def visit(node, source):
        if isinstance(node, dict):
            for key, item in node.items():
                child = source + '.' + key if source else key
                if key in ('warnings', 'errors'):
                    marker = (key, _identity(item))
                    if marker not in found:
                        found[marker] = {'kind': key, 'sources': [], 'items': copy.deepcopy(item)}
                    found[marker]['sources'].append(child)
                else:
                    visit(item, child)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f'{source}[{index}]')

    visit(value, '')
    return list(found.values())


_SLOT_KEYS = ('role', 'slot_index', 'source_material', 'source_material_key', 'source_material_id')


def _slot_groups(slots, *, applied=False):
    """Lossless for role/name/index selector matches and matched-slot counts.

    Component object_name is deliberately absent: it is NOT a material selector.
    Local validators still use the original component-level inventory, including
    those names, for overlapping assignment detection.
    """
    keys = _SLOT_KEYS + (('applied_material_id', 'applied_material') if applied else ())
    groups = {}
    unknown = []
    for slot in slots if isinstance(slots, list) else []:
        if not isinstance(slot, dict) or 'object_id' not in slot:
            unknown.append(copy.deepcopy(slot))
            continue
        fields = _pick(slot, keys)
        marker = _identity(fields)
        if marker not in groups:
            groups[marker] = {**fields, 'objects': {}}
        counts = groups[marker]['objects']
        identifier = slot['object_id']
        counts[identifier] = counts.get(identifier, 0) + 1
    result = list(groups.values())
    return result, unknown


def compact_material_binding(binding):
    """Group repeated source slots, preserving all exact selector alternatives."""
    if not isinstance(binding, dict):
        return copy.deepcopy(binding)
    result = _pick(binding, ('schema_version', 'status', 'requested_count', 'applied_slot_count',
                             'errors', 'warnings', 'role_source'))
    result['inventory_format'] = ('available_slot_groups: exact source role/name/index tuple; '
        'objects maps each expanded object_id to matching component-slot count. '
        'Different groups are alternative tuples, not a Cartesian product. '
        'Component names are omitted because they are not material selectors. '
        'Full component inventory and overlap validation remain local.')
    if 'available_slots' in binding:
        if not isinstance(binding['available_slots'], list):
            result['available_slot_groups'] = None
            result['inventory_status'] = 'unusable_source_inventory'
        else:
            result['available_slot_count'] = len(binding['available_slots'])
            result['available_slot_groups'], unknown = _slot_groups(binding['available_slots'])
            if unknown:
                result['unrecognized_slots'] = unknown
    if 'bindings' in binding:
        result['bindings'] = []
        for entry in binding['bindings']:
            item = _pick(entry, ('override_index', 'object_id', 'material_id', 'selector', 'status',
                                 'errors', 'warnings'))
            if 'matched_slots' in entry:
                item['matched_slot_count'] = len(entry['matched_slots'])
                item['matched_slot_groups'], unknown = _slot_groups(entry['matched_slots'], applied=True)
                if unknown:
                    item['unrecognized_slots'] = unknown
            result['bindings'].append(item)
    return result


def _compact_spatial(spatial):
    if not isinstance(spatial, dict):
        return copy.deepcopy(spatial)
    result = _pick(spatial, ('method', 'coordinate_origin', 'positive_error_means',
        'requires_inspection', 'reasons', 'thresholds', 'guidance', 'limitations',
        'measurement_stage', 'warnings', 'errors'))
    if 'per_view' in spatial:
        # These bounded 3x3 observations are current measurements, not candidates.
        result['per_view'] = {}
        for view_id, view in spatial['per_view'].items():
            item = _pick(view, ('status', 'usable_region_count', 'p90_minus_p10_error_stops',
                'median_absolute_error_stops', 'nonuniform_residual', 'warnings', 'errors'))
            if 'regions' in view:
                item['regions'] = [_pick(region, ('id', 'normalized_bounds_xyxy', 'reference_p65',
                    'render_p65', 'log2_luminance_error', 'status', 'warnings', 'errors'))
                    for region in view['regions']]
            result['per_view'][view_id] = item
    return result


def compact_lighting(metadata):
    """Current actual lighting and bounded-fit diagnostics; no search history."""
    if not isinstance(metadata, dict):
        return {}
    lighting = metadata.get('lighting', {})
    tone = metadata.get('tone_calibration', {})
    result = {'lighting': _pick(lighting, ('method', 'fallback_inferred', 'camera_fill',
        'emitters', 'base_lighting', 'limitations', 'exterior_proxy_transmission_hidden',
        'glazing_transport_approximations', 'visible_fixture_faces', 'applied_after_reference_fit',
        'warnings', 'errors'))}
    compact_tone = _pick(tone, ('status', 'correction_ev', 'max_correction_ev', 'initial_exposure',
        'final_exposure', 'view_transform', 'look', 'at_correction_bound', 'shared_across_views',
        'limitations', 'warnings', 'errors'))
    fit = tone.get('lighting_fit') if isinstance(tone, dict) else None
    if isinstance(fit, dict):
        compact_tone['lighting_fit'] = _pick(fit, ('status', 'objective_version',
            'severe_region_residuals', 'selected_power_at_bound', 'requires_non_power_inspection',
            'residual_guidance', 'shared_across_views', 'baseline_spatial_loss', 'selected_spatial_loss',
            'relative_loss_reduction', 'multiplier_bounds', 'exposure_correction_bounds',
            'applied_settings', 'ignored_controls', 'geometry_changed', 'material_changed',
            'per_camera_parameters', 'requires_independent_visual_review', 'limitations', 'warnings', 'errors'))
        # The same applied records appear in two locations in current renderers.
        if (result['lighting'].get('applied_after_reference_fit') == fit.get('applied_settings')
                and 'applied_settings' in fit):
            result['lighting'].pop('applied_after_reference_fit', None)
    has_final_spatial = isinstance(tone, dict) and isinstance(tone.get('final_spatial_diagnostics'), dict)
    if has_final_spatial:
        # The fit-pass grid can differ from the saved final PNG and is
        # superseded for review. Keep the final measurement only; preserve the
        # older field as a fallback for renderers without final diagnostics.
        compact_tone['final_spatial_diagnostics'] = _compact_spatial(tone['final_spatial_diagnostics'])
    elif isinstance(tone, dict) and 'spatial_diagnostics' in tone:
        compact_tone['spatial_diagnostics'] = _compact_spatial(tone['spatial_diagnostics'])
    result['tone_calibration'] = compact_tone
    return result


_PROJECTION_PART_KEYS = ('part_id', 'part_index', 'part_type', 'world_aabb_m',
    'camera_depth_range_m', 'projected_bbox_normalized', 'frame_clipped_bbox_normalized',
    'projected_bbox_in_frame_fraction', 'frame_bbox_coverage_fraction',
    'intersects_frame_before_occlusion', 'status', 'render_enabled',
    # Older local annotation revisions use these names; do not drop evidence.
    'in_frame', 'normalized_bbox_xyxy')


def compact_part_projection(projection, group_ids=None):
    """Columnar measured projections with exact original group/part identities.

    Optional group selection is explicit, with omitted counts. It must never be
    used as proof that omitted parts do not exist or that frustum hits are visible.
    All selected part indices remain original zero-based indices.
    """
    if not isinstance(projection, dict):
        return copy.deepcopy(projection)
    result = _pick(projection, ('schema_version', 'view_id', 'resolution_px', 'box_convention',
        'coordinate_origin', 'scope', 'occlusion_tested', 'part_count', 'status_counts', 'errors', 'warnings'))
    if 'parts' not in projection:
        return result
    source = projection['parts']
    grouped = {}
    omitted = 0
    for part in source:
        group_id = part.get('group_id')
        if group_ids is not None and group_id not in group_ids:
            omitted += 1
            continue
        grouped.setdefault(group_id, []).append(part)
    result['part_groups'] = []
    for group_id, parts in grouped.items():
        columns = [key for key in _PROJECTION_PART_KEYS if any(key in part for part in parts)]
        result['part_groups'].append({'group_id': group_id, 'columns': columns,
            'rows': [[copy.deepcopy(part.get(key)) for key in columns] for part in parts]})
    result['format'] = ('Rows follow columns; null means missing or unknown. Indices are original '
        'zero-based group indices. In-frame/frustum intersection is NOT visibility; '
        'no occlusion result is inferred. Full projections remain local.')
    result['omitted_part_count'] = omitted
    return result


def compact_per_view_geometry(evidence, group_ids=None):
    """Compact existing per-view evidence without changing pairing or warnings."""
    result = []
    for view in evidence:
        item = _pick(view, ('view_id', 'warnings', 'errors', 'support_checks'))
        if 'custom_part_projection' in view:
            item['custom_part_projection'] = compact_part_projection(view['custom_part_projection'], group_ids)
        result.append(item)
    return result


def compact_review_metadata(metadata, *, final_only=False):
    """Strict allowlist for review; never pass full provenance/render histories.

    A final assessment has no model-side edit, so its request omits selector
    alternatives used only to plan material edits and superseded fit-pass
    measurements. Applied assignments, diagnostics and final render evidence
    remain available; omitted alternatives are never evidence of a capability
    gap.
    """
    result = _pick(metadata, ('status', 'schema_version', 'scene_id', 'scene_type', 'camera_count',
        'asset_count', 'custom_group_count', 'mesh_count', 'curve_count', 'texture_count',
        'architecture', 'appearance_policy', 'quality_status'))
    result['context_version'] = CONTEXT_VERSION
    result['views'] = [_pick(view, ('id', 'width', 'height', 'warning_count'))
                       for view in metadata.get('views', [])]
    result['render'] = _pick(metadata.get('render'), ('engine', 'device', 'samples', 'max_edge',
        'view_transform', 'look', 'exposure', 'tone_match'))
    result.update(compact_lighting(metadata))
    if final_only and 'final_spatial_diagnostics' in result['tone_calibration']:
        result['tone_calibration'].pop('spatial_diagnostics', None)
    geometry = metadata.get('geometry_preflight')
    if isinstance(geometry, dict):
        item = _pick(geometry, ('status', 'room_interior_aabb_m', 'errors', 'warnings', 'scope', 'camera_checks'))
        item['objects_with_boundary_violations'] = [
            _pick(obj, ('object_id', 'world_aabb_m', 'structural_exemption', 'severe_boundary_violations'))
            for obj in geometry.get('objects', []) if obj.get('severe_boundary_violations')]
        if 'daylight_checks' in geometry:
            item['daylight_checks'] = _pick(geometry['daylight_checks'], ('errors', 'warnings', 'windows', 'scope'))
        result['geometry_preflight'] = item
    binding = metadata.get('material_binding', metadata.get('provenance', {}).get('material_binding'))
    if binding is not None:
        compact_binding = compact_material_binding(binding)
        if final_only and isinstance(compact_binding, dict):
            compact_binding.pop('available_slot_groups', None)
            compact_binding.pop('inventory_format', None)
            if 'available_slot_count' in compact_binding:
                compact_binding['inventory_detail'] = (
                    'Selector alternatives omitted for final assessment; absence is not evidence of unsupported materials.')
        result['material_binding'] = compact_binding
    if isinstance(metadata.get('exterior_context'), dict):
        result['exterior_context'] = _pick(metadata['exterior_context'], ('status', 'contexts', 'errors', 'warnings', 'scope'))
    # This catches failures nested in omitted or newer diagnostics too, including
    # identical messages repeated in provenance. It does not weaken local gates.
    result['all_diagnostic_messages'] = diagnostic_messages(metadata)
    result['omissions'] = ('Local provenance, file paths/hashes, search candidates and component-level '
        'slot dumps omitted. Complete diagnostics remain bound to the review locally; '
        'absence of an omitted field is not evidence of correctness.')
    return result
