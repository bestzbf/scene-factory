"""Deterministic scene-type corrections based on the normalized base inventory."""

from __future__ import annotations

import copy


def normalize_scene_type_inventory(spec):
    """Apply the high-confidence shop-to-office inventory rule.

    The model's response is never modified in place.  Counts are based on
    expanded base instances, so repeated groups contribute every instance to
    the evidence used for classification.
    """
    result = copy.deepcopy(spec)
    original = result.get('scene_type')
    counts = {'desk': 0, 'monitor': 0, 'computer': 0, 'monitor_or_computer': 0, 'chair': 0}
    for entry in result.get('base_scene', {}).get('objects', []):
        category = entry.get('category')
        repeat = entry.get('repeat', [1, 1])
        try:
            expanded = repeat[0] * repeat[1]
        except (IndexError, TypeError):
            expanded = 0
        if category in counts:
            counts[category] += expanded
        if category in ('monitor', 'computer'):
            counts['monitor_or_computer'] += expanded

    office_evidence = (original == 'shop' and counts['desk'] >= 2
                       and counts['monitor_or_computer'] >= 2
                       and counts['chair'] >= 2)
    if office_evidence:
        result['scene_type'] = 'office'
        if isinstance(result.get('base_scene'), dict):
            result['base_scene']['scene_type'] = 'office'
        reason = ('shop inventory has at least two expanded desks, two monitors/computers, '
                  'and two chairs; classify as office')
    else:
        reason = ('shop inventory does not meet the high-confidence office threshold'
                  if original == 'shop' else 'scene type is not shop; no inventory correction applied')

    record = {
        'from': original,
        'to': result.get('scene_type'),
        'counts': counts,
        'reason': reason,
    }
    return result, record
