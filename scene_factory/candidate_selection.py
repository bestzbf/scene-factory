"""Fail-closed selection of independently reviewed render candidates.

The batch runner can render several repair candidates.  A model's repair is
never assumed to be an improvement: this module compares only candidates that
have a fresh paired review and keeps the best reviewed candidate as the
incumbent.  The functions are deliberately pure so selection can be tested
without Blender, API calls, or image fixtures.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


DIMENSIONS = ("framing", "layout", "geometry", "materials", "lighting", "realism")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _score_maps(report: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return minimum per-view scores and the validated view score records.

    Invalid/missing review scores are not silently treated as zero.  A caller
    receives ``ValueError`` and therefore cannot accidentally certify an
    unscored candidate.
    """
    if not isinstance(report, Mapping) or not isinstance(report.get("per_view_scores"), list):
        raise ValueError("candidate review lacks per_view_scores")
    views: dict[str, dict[str, float]] = {}
    for item in report["per_view_scores"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("view_id"), str):
            raise ValueError("candidate review has an invalid view score")
        view_id = item["view_id"]
        if view_id in views or not isinstance(item.get("scores"), Mapping):
            raise ValueError("candidate review has duplicate or missing view scores")
        values: dict[str, float] = {}
        for dimension in DIMENSIONS:
            value = item["scores"].get(dimension)
            if not _finite_number(value) or not 0 <= value <= 100:
                raise ValueError(f"candidate review has an invalid {dimension} score")
            values[dimension] = float(value)
        views[view_id] = values
    if not views:
        raise ValueError("candidate review has no scored views")
    minima = {dimension: min(values[dimension] for values in views.values())
              for dimension in DIMENSIONS}
    return minima, views


def _spatial_metrics(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract optional final-render spatial error diagnostics.

    The diagnostics are evidence, not a replacement for the independent visual
    review.  Missing diagnostics remain missing so old renders can still be
    compared by their review scores.
    """
    if not isinstance(metadata, Mapping):
        return {"median_absolute_error_stops": None, "spread_stops": None, "per_view": {}}
    # Calibration reports meter low-resolution fitting previews; those may not
    # be identical to the rendered candidate the reviewer assessed.
    tone = metadata.get("final_spatial_diagnostics", {})
    spatial = tone.get("spatial_diagnostics", tone) if isinstance(tone, Mapping) else {}
    per_view = spatial.get("per_view", {}) if isinstance(spatial, Mapping) else {}
    medians, spreads, measurements = [], [], {}
    if isinstance(per_view, Mapping):
        for view_id, value in per_view.items():
            if not isinstance(value, Mapping):
                continue
            median = value.get("median_absolute_error_stops")
            spread = value.get("p90_minus_p10_error_stops")
            if _finite_number(median):
                medians.append(float(median))
            if _finite_number(spread):
                spreads.append(float(spread))
            measurements[view_id] = {
                'median_absolute_error_stops': float(median) if _finite_number(median) else None,
                'spread_stops': float(spread) if _finite_number(spread) else None}
    return {"median_absolute_error_stops": max(medians) if medians else None,
            "spread_stops": max(spreads) if spreads else None, "per_view": measurements}


def candidate_metrics(report: Mapping[str, Any], gate: Mapping[str, Any] | None = None,
                      metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a deterministic, review-only metric record for one candidate."""
    minima, views = _score_maps(report)
    values = [score for item in views.values() for score in item.values()]
    worst = min(values)
    mean = sum(values) / len(values)
    # The minimum score protects the weakest view/dimension; the mean makes
    # small, coherent improvements visible without hiding a severe regression.
    quality = 0.7 * worst + 0.3 * mean
    spatial = _spatial_metrics(metadata)
    accepted = bool(isinstance(gate, Mapping) and gate.get("accepted") is True)
    issues = report.get("issues", [])
    asset_gaps = report.get("asset_gaps", [])
    if not isinstance(issues, list) or not isinstance(asset_gaps, list):
        raise ValueError("candidate review has malformed issues or asset gaps")
    major_count = sum(1 for issue in issues if isinstance(issue, Mapping)
                      and issue.get("severity") in ("major", "blocking"))
    return {"minimum_by_dimension": minima, "per_view_scores": views, "worst_score": worst,
            "mean_score": mean, "quality_score": quality,
            "spatial": spatial, "gate_accepted": accepted,
            "view_ids": sorted(views), "major_issue_count": major_count,
            "asset_gap_count": len(asset_gaps)}


def _delta_map(challenger: Mapping[str, float], incumbent: Mapping[str, float]) -> dict[str, float]:
    return {key: float(challenger[key]) - float(incumbent[key]) for key in DIMENSIONS}


def compare_candidates(incumbent: Mapping[str, Any] | None,
                       challenger: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two independently reviewed candidate metric records.

    Accepted gates are terminal.  Otherwise a candidate must improve the
    protected worst score, or improve the joint quality score without lowering
    that worst score.  A modest photometric improvement can also win when it
    does not trade away more than half a score point.  Large per-dimension,
    issue, or photometric regressions are rejected.  Ties retain the incumbent
    for deterministic resume behavior.
    """
    if not isinstance(challenger, Mapping):
        raise ValueError("challenger metrics are required")
    if isinstance(incumbent, Mapping) and incumbent.get("gate_accepted") is True:
        return {"selected": False, "reason": "accepted_incumbent_terminal", "deltas": None,
                "regressions": [], "incumbent": incumbent, "challenger": challenger}
    if challenger.get("gate_accepted") is True:
        return {"selected": True, "reason": "accepted_candidate_terminal", "deltas": None,
                "regressions": [], "incumbent": incumbent, "challenger": challenger}
    if incumbent is None:
        return {"selected": True, "reason": "first_reviewed_candidate", "deltas": None,
                "regressions": [], "incumbent": None, "challenger": challenger}
    try:
        incumbent_dims = incumbent["minimum_by_dimension"]
        challenger_dims = challenger["minimum_by_dimension"]
        deltas = _delta_map(challenger_dims, incumbent_dims)
    except (KeyError, TypeError):
        raise ValueError("candidate metrics lack per-dimension scores")
    if incumbent.get("view_ids") != challenger.get("view_ids"):
        raise ValueError("candidate views differ; joint comparison is not valid")
    regressions = [f"{key}:{value:.2f}" for key, value in deltas.items() if value < -3.0]
    for view_id in incumbent.get("view_ids", []):
        old = incumbent.get("per_view_scores", {}).get(view_id, {})
        new = challenger.get("per_view_scores", {}).get(view_id, {})
        for dimension in DIMENSIONS:
            if dimension in old and dimension in new and new[dimension] - old[dimension] < -3.0:
                regressions.append(f"{view_id}:{dimension}:{new[dimension]-old[dimension]:.2f}")
    major_delta = int(challenger.get("major_issue_count", 0)) - int(incumbent.get("major_issue_count", 0))
    gaps_delta = int(challenger.get("asset_gap_count", 0)) - int(incumbent.get("asset_gap_count", 0))
    if major_delta > 0 or gaps_delta > 0:
        regressions.extend(([f"major_issue_count:+{major_delta}"] if major_delta > 0 else [])
                           + ([f"asset_gap_count:+{gaps_delta}"] if gaps_delta > 0 else []))
    # A quality score may not improve by sacrificing a dominant dimension.
    if regressions:
        reason = 'new_major_issues_or_asset_gaps' if major_delta > 0 or gaps_delta > 0 else 'dimension_regression'
        return {"selected": False, "reason": reason, "deltas": deltas,
                "regressions": regressions, "incumbent": incumbent, "challenger": challenger}
    worst_delta = float(challenger["worst_score"]) - float(incumbent["worst_score"])
    quality_delta = float(challenger["quality_score"]) - float(incumbent["quality_score"])
    old_spatial = incumbent.get("spatial", {})
    new_spatial = challenger.get("spatial", {})
    old_median, new_median = old_spatial.get("median_absolute_error_stops"), new_spatial.get("median_absolute_error_stops")
    old_spread, new_spread = old_spatial.get("spread_stops"), new_spatial.get("spread_stops")
    spatial_delta = None
    spread_delta = None
    if _finite_number(old_median) and _finite_number(new_median):
        spatial_delta = float(old_median) - float(new_median)  # positive is better
    if _finite_number(old_spread) and _finite_number(new_spread):
        spread_delta = float(old_spread) - float(new_spread)
    regional_regressions = []
    for view_id, values in old_spatial.get('per_view', {}).items():
        next_values = new_spatial.get('per_view', {}).get(view_id, {})
        for key, tolerance in (('median_absolute_error_stops', .35), ('spread_stops', .5)):
            old, new = values.get(key), next_values.get(key)
            if _finite_number(old) and _finite_number(new) and new-old > tolerance:
                regional_regressions.append(f'{view_id}:{key}:+{new-old:.2f}')
    # If measured brightness became materially worse, do not let a language
    # model's subjective score select the candidate.
    if (regional_regressions or (spatial_delta is not None and spatial_delta < -0.35)
            or (spread_delta is not None and spread_delta < -0.5)):
        return {"selected": False, "reason": "photometric_regression", "deltas": deltas,
                "regressions": regional_regressions + ([f"median_absolute_error_stops:{-spatial_delta:.2f}"]
                                if spatial_delta is not None and spatial_delta < -0.35 else [])
                               + ([f"spread_stops:{-spread_delta:.2f}"]
                                  if spread_delta is not None and spread_delta < -0.5 else []),
                "incumbent": incumbent, "challenger": challenger}
    selected = ((worst_delta >= 1.0 and quality_delta >= -0.5)
                or (worst_delta >= 0.0 and quality_delta >= 1.0)
                or (spatial_delta is not None and spatial_delta >= 0.25 and worst_delta >= -0.5
                    and quality_delta >= -0.5))
    if selected:
        reason = "joint_review_improved"
        if spatial_delta is not None and spatial_delta >= 0.25 and quality_delta < 1.0:
            reason = "photometric_improvement"
    else:
        reason = "no_objective_improvement"
    return {"selected": selected, "reason": reason, "deltas": deltas,
            "worst_score_delta": worst_delta, "quality_score_delta": quality_delta,
            "spatial_median_delta": spatial_delta, "spatial_spread_delta": spread_delta,
            "regressions": regressions, "incumbent": incumbent, "challenger": challenger}


def select_candidate(incumbent: Mapping[str, Any] | None, challenger: Mapping[str, Any],
                     *, iteration: str | None = None) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Return the winning candidate and a serializable comparison record."""
    decision = compare_candidates(incumbent.get("metrics") if incumbent else None,
                                  challenger.get("metrics", challenger))
    if decision["selected"]:
        winner = challenger
    else:
        winner = incumbent if incumbent is not None else challenger
    record = {key: value for key, value in decision.items() if key not in ("incumbent", "challenger")}
    if iteration is not None:
        record["challenger_iteration"] = iteration
    return winner, record
