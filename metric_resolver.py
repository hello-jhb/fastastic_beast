"""
metric_resolver.py — Phase 1 candidate ranking + schema validation.

Given a list of candidates from scan_workbook_for_candidates() and the metric's
schema (unit, scale, period, valid range, preferred sheets), this module:

  1. Filters out candidates that fail schema constraints
  2. Auto-corrects scale issues (e.g., value in $000s → multiply by 1,000)
  3. Ranks the survivors by preferred-sheet priority + signal quality
  4. Returns a verified metric record OR a candidate-pool record

The output shape is the new SSOT metric format:
  {
    raw_value, normalized_value, display_value,
    unit, scale, period,
    source_sheet, source_cell,
    confidence (extractor confidence: exact/high/medium/partial),
    status (verified | candidate_pool | suspicious | missing),
    validation_notes (list of strings),
    candidates  (full list, retained for resolver / audit)
  }

In Phase 2 a GPT resolver will be inserted between step 1 and step 3 to
disambiguate when multiple candidates survive validation.
"""
from __future__ import annotations
from typing import Any


RESOLVER_VERSION = "phase2.v1"


# ---------------------------------------------------------------------------
# Schema-driven validation
# ---------------------------------------------------------------------------

def _validate_against_range(value, schema: dict) -> tuple[bool, str | None]:
    """
    Return (passes, scale_correction_factor or None).

    If value is in range → passes=True.
    If value × 1000 is in range → passes=True, returns 1000 (auto-scale).
    If value × 1_000_000 is in range → passes=True, returns 1_000_000.
    Otherwise → passes=False.
    """
    # Non-numeric units (text, date) don't have numeric ranges — accept as-is
    unit = schema.get("unit")
    if unit in ("text", "date"):
        return True, None

    rmin = schema.get("range_min")
    rmax = schema.get("range_max")
    if rmin is None and rmax is None:
        return True, None  # no range constraint

    def _in_range(v):
        if rmin is not None and v < rmin:
            return False
        if rmax is not None and v > rmax:
            return False
        return True

    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, None

    if _in_range(v):
        return True, None
    if _in_range(v * 1_000):
        return True, "1000"
    if _in_range(v * 1_000_000):
        return True, "1000000"
    return False, None


def _preferred_sheet_score(sheet_name: str, preferred_sheets: list[str]) -> int:
    """
    Lower score = better match. Returns:
      0..N-1: index in preferred_sheets (first preferred → 0)
      999:    not in preferred list
    """
    if not preferred_sheets:
        return 100
    sn = sheet_name.lower()
    for i, pref_keyword in enumerate(preferred_sheets):
        if pref_keyword.lower() in sn:
            return i
    return 999


def _format_display(value, unit: str | None, scale: str | None) -> str:
    """Format a value for narrative use given its unit/scale."""
    if value is None:
        return "—"
    if unit == "text":
        return str(value)
    if unit == "date":
        import datetime as _dt
        if isinstance(value, _dt.datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, _dt.date):
            return value.isoformat()
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if unit == "USD":
        if abs(v) >= 1_000_000:
            return f"${v/1_000_000:,.2f}M"
        if abs(v) >= 1_000:
            return f"${v/1_000:,.0f}K"
        return f"${v:,.2f}"
    if unit == "ratio":
        if abs(v) < 1.5:           # treat as fraction → percentage
            return f"{v*100:.2f}%"
        return f"{v:,.2f}x"
    if unit == "percent":
        return f"{v:.2f}%"
    if unit == "months":
        if v >= 12:
            return f"{v:.0f} months ({v/12:.1f} yr)"
        return f"{v:.0f} months"
    if unit == "years":
        return f"{v:.1f} years"
    if unit == "count":
        return f"{int(v):,}"
    if unit == "sf":
        return f"{int(v):,} SF"
    return f"{v:,.2f}"


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_metric(metric: dict, candidates: list[dict]) -> dict:
    """
    Apply schema validation + preferred-sheet ranking to pick the best candidate.

    Returns a SSOT-shaped metric record.
    """
    notes: list[str] = []
    preferred_sheets = metric.get("preferred_sheets", []) or []
    unit  = metric.get("unit")
    scale = metric.get("scale")
    period = metric.get("period")

    if not candidates:
        return _make_record(
            metric=metric, candidate=None, normalized_value=None,
            status="missing", notes=["No candidates found in any scanned sheet."],
            candidates=[],
        )

    # Score every candidate
    scored = []
    for cand in candidates:
        c_notes: list[str] = []

        # Range / scale validation
        passes_range, scale_correction = _validate_against_range(cand["value"], metric)
        if not passes_range:
            c_notes.append(
                f"value {cand['value']} outside range "
                f"[{metric.get('range_min')}, {metric.get('range_max')}]"
            )
        elif scale_correction:
            c_notes.append(f"auto-scaled by {scale_correction} (raw was likely in 000s/M)")

        # Preferred sheet ranking
        pref_score = _preferred_sheet_score(cand["sheet"], preferred_sheets)

        # Annotate the candidate dict itself (so Phase 2 resolver can read it)
        cand["passes_validation"] = passes_range
        cand["scale_correction"]  = scale_correction
        cand["pref_score"]        = pref_score

        scored.append({
            "candidate":         cand,
            "passes_validation": passes_range,
            "scale_correction":  scale_correction,
            "pref_score":        pref_score,
            "notes":             c_notes,
        })

    # Filter to validation-passing candidates first
    passing = [s for s in scored if s["passes_validation"]]
    pool = passing or scored  # fallback to all candidates if none pass

    # Sort by:
    #   1. preferred_sheet_score (lower is better)
    #   2. sheet_tier (lower is better)
    #   3. extractor confidence
    _CONF_TIER = {"exact": 0, "high": 1, "medium": 2, "partial": 3}
    pool.sort(key=lambda s: (
        s["pref_score"],
        s["candidate"].get("sheet_tier", 99),
        _CONF_TIER.get(s["candidate"]["confidence"], 9),
        -s["candidate"].get("label_ratio", 0),
    ))

    top = pool[0]
    cand = top["candidate"]
    raw_value = cand["value"]

    # Apply scale correction if needed
    if top["scale_correction"] == "1000":
        normalized_value = float(raw_value) * 1000
    elif top["scale_correction"] == "1000000":
        normalized_value = float(raw_value) * 1_000_000
    else:
        normalized_value = raw_value

    # Determine status
    if top["passes_validation"]:
        if len(passing) > 1:
            # Multiple candidates passed schema — needs Phase 2 resolver
            status = "candidate_pool"
            notes.append(
                f"{len(passing)} candidates passed schema validation; "
                "top one taken by preferred-sheet rank. "
                "Phase 2 resolver will disambiguate."
            )
        else:
            status = "verified"
    else:
        status = "suspicious"
        notes.append("No candidate passed schema validation. Top candidate by rank used; may be wrong.")

    notes.extend(top["notes"])

    return _make_record(
        metric=metric,
        candidate=cand,
        normalized_value=normalized_value,
        status=status,
        notes=notes,
        candidates=[s["candidate"] for s in scored],
    )


def _make_record(
    metric: dict,
    candidate: dict | None,
    normalized_value,
    status: str,
    notes: list[str],
    candidates: list[dict],
) -> dict:
    unit  = metric.get("unit")
    scale = metric.get("scale")
    period = metric.get("period")
    return {
        "metric_id":         metric["metric_id"],
        "metric_name":       metric["metric_name"],
        "raw_value":         candidate["value"] if candidate else None,
        "normalized_value":  normalized_value,
        "display_value":     _format_display(normalized_value, unit, scale),
        "unit":              unit,
        "scale":             scale,
        "period":            period,
        "source_sheet":      candidate["sheet"] if candidate else None,
        "source_cell":       candidate["value_cell"] if candidate else None,
        "sheet_tier":        candidate.get("sheet_tier") if candidate else None,
        "extractor_confidence": candidate["confidence"] if candidate else None,
        "status":            status,
        "validation_notes":  notes,
        "candidates":        candidates,  # full list, for audit / Phase 2 resolver
        "in_bounded_list":   metric.get("in_bounded_list", False),
    }


# ---------------------------------------------------------------------------
# Cache key — Phase 1 versioning
# ---------------------------------------------------------------------------

def make_cache_key(
    file_hash: str,
    catalog_version: str,
    extractor_version: str | None = None,
    resolver_version: str | None = None,
) -> str:
    """
    Build a versioned cache key. Any version bump invalidates old cached results.
    """
    import hashlib
    if extractor_version is None:
        from flexible_extractor import EXTRACTOR_VERSION as ev
        extractor_version = ev
    if resolver_version is None:
        resolver_version = RESOLVER_VERSION
    composite = f"{file_hash}|{catalog_version}|{extractor_version}|{resolver_version}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()
