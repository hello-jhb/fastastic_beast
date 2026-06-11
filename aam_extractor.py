"""
aam_extractor.py — Stage-1 focused extraction of the Audit Appendix Metrics.

This is the redesigned first pass (2026-06-10). It replaces the broad, automatic
GPT sweep (5 section reads + insight pass + comprehension review over all 109
catalog metrics) with a tight, AAM-scoped pass:

    Workbook Mapper (classify_sheets, GPT call #1)
        |  role map
    Deterministic resolve (proximity scan + resolve_metric, scoped to ~20 AAM ids)
        |  fills what it can confidently, with schema + source-hierarchy validation
    Focused GPT gap-fill (ONE batched call, GPT call #2 -- only the gaps)
        |
    AAM records  ->  Audit Appendix  ->  human verification gate

The deterministic half runs with NO API key. Both GPT calls are skipped silently
when llm_available() is False, so the appendix still renders (with blanks the
user can later fill on demand).

This is NOT a new extractor. It is a scope + orchestration layer that reuses the
existing engine wholesale: flexible_extractor primitives, metric_resolver
ranking, and sheet_classifier -- restricted to the AAM.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from aam import aam_metrics
from metric_catalog import load_metric_catalog
from flexible_extractor import (
    scan_workbook_for_candidates,
    extract_raw_labeled_pairs,
    sheet_priority_tier,
)
from metric_resolver import resolve_metric
from scenarios._llm import client, MODEL_FAST, llm_available

log = logging.getLogger("fb.aam_extractor")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[fb.aam] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

AAM_EXTRACTOR_VERSION = "2026-06-10.1"

# Deterministic statuses that warrant a focused GPT gap-fill attempt.
_GAP_STATUSES = {"missing", "candidate_pool", "suspicious"}

# Max labeled pairs to hand the focused GPT read (keeps one call cheap).
_MAX_PAIRS = 450


def extract_aam(
    file_path: str | Path,
    layer: str = "underwriting",
    sheet_classification: dict[str, dict] | None = None,
    use_gpt_gap_fill: bool = True,
) -> dict[str, Any]:
    """
    Run the focused Stage-1 AAM extraction.

    Args:
      file_path:            workbook to read.
      layer:                SSOT layer (kept for parity / future per-layer AAM).
      sheet_classification: pre-computed role map; if None and an API key is
                            present, classify_sheets is called once (GPT #1).
      use_gpt_gap_fill:     run the single focused GPT gap-fill (GPT #2) for AAM
                            fields the deterministic pass left missing/ambiguous.

    Returns:
      { metric_name: resolver_record }  for every AAM field, in AAM order.
    """
    file_path = Path(file_path)
    catalog = load_metric_catalog()
    aam = aam_metrics(catalog)

    # --- Step 1: Workbook Mapper (GPT call #1, batched, cheap) ----------------
    classification, sheet_tier_map = _build_tier_map(file_path, sheet_classification)

    # --- Step 2: Deterministic resolve, scoped to the AAM --------------------
    candidates_by_metric = scan_workbook_for_candidates(file_path, aam, sheet_tier_map)
    records: dict[str, Any] = {}
    for m in aam:
        cands = candidates_by_metric.get(m["metric_id"], [])
        records[m["metric_name"]] = resolve_metric(m, cands)

    det_counts = _status_counts(records)
    log.info(
        "AAM deterministic pass for %s -- %s",
        file_path.name, ", ".join(f"{k}={v}" for k, v in sorted(det_counts.items())),
    )

    # --- Step 3: Focused GPT gap-fill (GPT call #2, single batched call) ------
    if use_gpt_gap_fill and llm_available():
        gaps = [m for m in aam if records[m["metric_name"]]["status"] in _GAP_STATUSES]
        if gaps:
            filled = _focused_gap_fill(file_path, gaps, records)
            log.info(
                "AAM focused GPT gap-fill for %s -- attempted %d, filled %d",
                file_path.name, len(gaps), filled,
            )

    # --- Step 4: AAM-scoped normalization (representation fixes) --------------
    # The audit appendix must show the human the RIGHT number to verify. These
    # transforms fix how an extracted value is represented; they do not invent
    # facts. Cross-metric derivation (e.g. Exit Date = Purchase + Hold) is left
    # to post-verification reconciliation so it never derives off an unverified
    # input the user is about to correct.
    _normalize_aam(records)

    # Drive NOI from the pricing identity (NOI = Price × Cap) rather than trust a
    # cell-matched NOI, which grabs the wrong period/column. Runs after blank-fill
    # too (via fill_aam_blanks) so newly-filled price/cap inputs feed the derive.
    _derive_noi_from_pricing(records)

    return records


def fill_aam_blanks(file_path: str | Path, records: dict[str, Any]) -> int:
    """
    Run the focused GPT gap-fill (GPT #2) over the blank/ambiguous rows of an
    EXISTING records dict (produced by a prior deterministic extract_aam).

    This is the on-demand "Fill blanks with GPT" action: one batched call over
    only the gaps, re-validated through resolve_metric, then re-normalized.
    Mutates `records` in place; returns the number of fields filled. Silently
    no-ops (returns 0) when no API key is available.
    """
    if not llm_available():
        return 0
    catalog = load_metric_catalog()
    aam = aam_metrics(catalog)
    gaps = [
        m for m in aam
        if records.get(m["metric_name"], {}).get("status") in _GAP_STATUSES
    ]
    if not gaps:
        return 0
    filled = _focused_gap_fill(Path(file_path), gaps, records)
    _normalize_aam(records)
    # Re-derive NOI now that blank-fill may have supplied missing price/cap inputs.
    _derive_noi_from_pricing(records)
    return filled


def _build_tier_map(
    file_path: Path,
    classification: dict[str, dict] | None,
) -> tuple[dict[str, dict], dict[str, int] | None]:
    """
    Resolve the content-role classification + per-sheet tier map.

    Reuses a caller-supplied classification when given; otherwise runs
    classify_sheets once (GPT #1). Returns ({}, None) gracefully when no API key
    is available, so the deterministic scan falls back to name-based tiers.
    """
    classification = classification or {}
    if not classification and llm_available():
        try:
            from sheet_classifier import classify_sheets
            classification = classify_sheets(file_path) or {}
        except Exception as e:
            log.error("Sheet classification failed for %s: %s", file_path.name, e)
            classification = {}

    if not classification:
        return {}, None

    try:
        import openpyxl
        from sheet_classifier import effective_tier
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        sheet_tier_map = {
            name: effective_tier(name, sheet_priority_tier(name), classification)
            for name in wb.sheetnames
        }
        wb.close()
        return classification, sheet_tier_map
    except Exception as e:
        log.error("Tier-map build failed for %s: %s", file_path.name, e)
        return classification, None


def _focused_gap_fill(
    file_path: Path,
    gaps: list[dict],
    records: dict[str, Any],
) -> int:
    """
    One batched GPT call over ALL gap metrics. Each returned hit is wrapped as a
    resolver candidate and re-validated through resolve_metric, so schema bounds
    and source-hierarchy still apply to GPT-sourced values. Mutates `records`
    in place; returns the number of fields filled.
    """
    pairs = extract_raw_labeled_pairs(file_path)
    if not pairs:
        return 0

    result = _aam_gpt_call(gaps, pairs)
    if not result:
        return 0

    filled = 0
    for m in gaps:
        hit = result.get(m["metric_id"])
        if not hit:
            continue
        value = hit.get("value")
        if value in (None, "", "-", "—"):
            continue

        sheet = hit.get("sheet") or ""
        candidate = {
            "metric_id":     m["metric_id"],
            "metric_name":   m["metric_name"],
            "value":         value,
            "source_file":   file_path.name,
            "sheet":         sheet,
            "sheet_tier":    sheet_priority_tier(sheet),
            "label_cell":    hit.get("cell"),
            "value_cell":    hit.get("cell"),
            "matched_alias": "(AAM focused read)",
            "confidence":    "medium",
            "label_ratio":   1.0,
            "match_method":  "aam_focused",
        }
        prev_status = records[m["metric_name"]]["status"]
        rec = resolve_metric(m, [candidate])
        # Only accept the GPT value if it actually validated to something usable.
        # Reject "suspicious" too: that means the value failed schema-range
        # validation (e.g. GPT grabbed the $192M purchase-price cell for Total SF,
        # which is far outside the SF range). A blank the human can fill is better
        # than a confident-looking out-of-range number. The field keeps its prior
        # (missing/blank) state rather than absorbing the bad fill.
        if rec["status"] in ("missing", "suspicious"):
            continue
        rec.setdefault("validation_notes", []).insert(
            0,
            f"Filled by focused AAM GPT read (deterministic pass left it "
            f"'{prev_status}').",
        )
        rec["_via_aam_gpt"] = True
        records[m["metric_name"]] = rec
        filled += 1

    return filled


_AAM_SYSTEM_PROMPT = """\
You are a real estate analyst reading an underwriting workbook. You are given a
flat list of labeled cells (sheet, cell, label, value) and a short list of
specific metrics to find. Find ONLY the metrics requested. For each one, return
the single best cell: its value, the exact sheet name, and the cell reference.

Rules:
- Use the value EXACTLY as it appears in the cell (do not reformat or convert).
- Prefer summary / inputs / assumptions / debt sheets over schedules or comps.
- If you cannot find a metric with reasonable confidence, OMIT it (do not guess).
- Confidence is "high", "medium", or "low".

Return ONLY JSON, no prose, no code fences:
{
  "<metric_id>": {"value": <number or string>, "sheet": "<exact sheet>",
                  "cell": "<A1>", "confidence": "high|medium|low"},
  ...
}
"""


def _aam_gpt_call(gaps: list[dict], pairs: list[dict]) -> dict[str, dict]:
    """Single focused GPT call. Returns { metric_id: {value, sheet, cell, confidence} }."""
    # Prefer pairs on lower-tier (more authoritative) sheets, keep the cap.
    pairs_sorted = sorted(pairs, key=lambda p: p.get("sheet_tier", 99))[:_MAX_PAIRS]
    pair_lines = [
        f"{p['sheet']}!{p['cell']}  {p['label']} = {p['value']}"
        for p in pairs_sorted
    ]

    metric_lines = []
    for m in gaps:
        rng = f"[{m.get('range_min')}, {m.get('range_max')}]"
        metric_lines.append(
            f"- {m['metric_id']} ({m['metric_name']}): "
            f"{m.get('definition', '')[:120]} "
            f"| unit={m.get('unit', '?')} | valid range={rng}"
        )

    user_msg = (
        "METRICS TO FIND:\n" + "\n".join(metric_lines)
        + "\n\nLABELED CELLS:\n" + "\n".join(pair_lines)
    )

    try:
        from knowledge_store import with_active_rules
        system_content = with_active_rules(_AAM_SYSTEM_PROMPT, ["metric_resolution", "validation"])
        response = client.chat.completions.create(
            model=MODEL_FAST,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as e:
        log.error("AAM focused read JSON parse failed: %s", e)
        return {}
    except Exception as e:
        log.error("AAM focused read API call failed: %s", e)
        return {}


def _as_num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_date(v: Any):
    import datetime as _dt
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return _dt.datetime.strptime(v[:19], fmt).date()
            except ValueError:
                continue
    return None


def _derived_hold_years(records: dict[str, Any]) -> float | None:
    """Years between Purchase Date and Exit Date, if both are present."""
    pd_rec, ed_rec = records.get("Purchase Date"), records.get("Exit Date")
    if not pd_rec or not ed_rec:
        return None
    a = _to_date(pd_rec.get("normalized_value") or pd_rec.get("raw_value"))
    b = _to_date(ed_rec.get("normalized_value") or ed_rec.get("raw_value"))
    if a and b:
        return abs((b - a).days) / 365.25
    return None


def _interpret_hold_value(v: float, derived: float | None) -> tuple[float, str] | None:
    """
    Decide whether a raw Hold Period value should be read as months and
    converted to years. Returns (years, note) if a conversion is warranted,
    else None (leave as-is).

      - With dates: pick the interpretation (years vs months) closest to the
        Purchase->Exit span. Disambiguates 12 / 18 / 24 correctly.
      - Without dates: fall back to the >24 month heuristic.
    """
    if derived is not None and derived > 0:
        fits_years = abs(v - derived)
        fits_months = abs(v / 12.0 - derived)
        if fits_months + 1e-9 < fits_years:
            years = round(v / 12.0, 1)
            return years, (
                f"Hold Period {v:g} read as MONTHS via date cross-check "
                f"(Purchase->Exit ~ {derived:.1f}y; {v:g}/12 ~ {years:.1f}y fits better "
                f"than {v:g}y)."
            )
        return None  # v already fits as years
    if 24 < v <= 360:
        years = round(v / 12.0, 1)
        return years, (
            f"Pattern fired: hold_period_gt_24_means_months. Normalized {v:g} "
            f"months -> {years:.1f} years (no dates available to cross-check)."
        )
    return None


def _normalize_aam(records: dict[str, Any]) -> None:
    """
    Representation-only normalization, scoped to AAM fields, applied in place.

    Currently handles the two transforms whose raw form misleads a verifier:
      - Hold Period stored in MONTHS -> years (the active
        `hold_period_gt_24_means_months` rule, mirrored from reconciliation).
      - Equity Multiple displayed as a multiplier ("1.40x"), not a percentage
        (the shared `ratio` formatter renders sub-1.5 ratios as %, which is
        correct for cap rates / LTV but wrong for an equity multiple).
    """
    # Hold Period: months -> years. Prefer a date cross-check (Purchase + Exit
    # dates) which disambiguates the ambiguous low values (12 / 18 / 24) the raw
    # >24 threshold can't; fall back to the threshold only when no dates exist.
    hp = records.get("Hold Period")
    if hp and hp.get("status") != "missing":
        v = _as_num(hp.get("normalized_value"))
        if v is None:
            v = _as_num(hp.get("raw_value"))
        if v is not None and v > 0:
            derived = _derived_hold_years(records)
            interp = _interpret_hold_value(v, derived)
            if interp:
                years, note = interp
                hp["normalized_value"] = years
                hp["display_value"] = f"{years:.1f} years"
                hp.setdefault("validation_notes", []).insert(0, note)

            # Sanity cross-check: if Purchase->Exit dates derive a hold the
            # extracted value disagrees with under BOTH the years AND the months
            # reading (even after conversion), the extracted cell is probably the
            # wrong one (e.g. a "Year 1" index, not the hold). Flag suspicious so
            # the human corrects it instead of trusting a wrong number. We do NOT
            # overwrite the value (dates may be unverified) — only surface it.
            if derived is not None and derived > 0:
                final_v = _as_num(hp.get("normalized_value")) or v
                best_err = min(
                    abs(final_v - derived),
                    abs(v - derived),
                    abs(v / 12.0 - derived),
                )
                if best_err / derived > 0.2:
                    hp["status"] = "suspicious"
                    hp.setdefault("validation_notes", []).insert(
                        0,
                        f"Hold Period {v:g} disagrees with the Purchase->Exit "
                        f"span (~{derived:.1f} years). The extracted cell may be "
                        f"wrong (e.g. a period index). Suggested: "
                        f"{derived:.1f} years."
                    )

    # Equity Multiple: always render as a multiplier.
    em = records.get("Equity Multiple")
    if em and em.get("status") != "missing":
        v = _as_num(em.get("normalized_value"))
        if v is not None:
            em["display_value"] = f"{v:.2f}x"


def _by_id(records: dict[str, Any], metric_id: str) -> dict[str, Any] | None:
    """Find an AAM record by metric_id (records are keyed by metric_name)."""
    for rec in records.values():
        if rec.get("metric_id") == metric_id:
            return rec
    return None


def _derive_noi_from_pricing(
    records: dict[str, Any],
    skip_ids: set[str] | None = None,
) -> None:
    """
    Drive NOI from the cap-rate identity (Price = NOI / Cap Rate), in place.

    An analyst reads NOI OUT of pricing, not the other way around: cell-matching
    grabs whichever NOI column it finds first (usually the exit/stabilized one,
    the higher number), which is why going-in and exit NOI collided on one cell.
    Deriving from price × cap is both more reliable and ties the appendix's NOI
    to the price and cap rate the human is verifying right beside it.

        Going-in NOI = Purchase Price × Going-in Cap Rate
        Exit NOI     = Exit Value     × Exit Cap Rate

    Best-effort and self-checking:
      - derives only when BOTH pricing inputs resolved to a usable value;
      - the COMPUTED NOI is range-checked against its own catalog schema, so a
        bad input (e.g. a cap that actually matched the NOI cell) yields an
        out-of-range product and is rejected rather than written;
      - a NOI whose id is in `skip_ids` is left untouched — used at confirm time
        when the human explicitly corrected the NOI itself (their value wins);
      - when a derived value disagrees with the extracted one by >10%, the note
        flags the extracted cell as the likely wrong period/column.
    """
    from metric_resolver import _format_display

    skip_ids = skip_ids or set()
    schema_by_id = {m["metric_id"]: m for m in load_metric_catalog()}

    # Only derive from a pricing input that actually validated. A 'suspicious'
    # value failed schema-range (e.g. BAC's going-in cap matched the $15.6M NOI
    # cell, not the 8.17% rate cell, before GPT fill) — feeding it would produce
    # a nonsense NOI. Better to leave the extracted NOI until the input is fixed.
    _USABLE = {"verified", "derived", "candidate_pool", "inferred"}

    def _num(rec: dict | None) -> float | None:
        if not rec or rec.get("status") not in _USABLE:
            return None
        v = _as_num(rec.get("normalized_value"))
        return v if (v is not None and v > 0) else None

    def _in_range(metric_id: str, value: float) -> bool:
        schema = schema_by_id.get(metric_id, {})
        lo, hi = schema.get("range_min"), schema.get("range_max")
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
        return True

    # (derived NOI id, price-input id, cap-input id, human-readable formula)
    rules = [
        ("net_operating_income_noi", "purchase_price", "going_in_cap_rate",
         "Purchase Price × Going-in Cap Rate"),
        ("exit_noi", "exit_value_terminal_value", "exit_cap_rate",
         "Exit Value × Exit Cap Rate"),
    ]
    for noi_id, price_id, cap_id, formula in rules:
        if noi_id in skip_ids:
            continue
        noi = _by_id(records, noi_id)
        if noi is None:
            continue
        price = _num(_by_id(records, price_id))
        cap = _num(_by_id(records, cap_id))
        if price is None or cap is None:
            continue  # missing a pricing input — leave the extracted NOI as-is

        derived = price * cap
        if not _in_range(noi_id, derived):
            continue  # nonsense product (bad input) — don't overwrite the NOI
        prev = _as_num(noi.get("normalized_value"))
        noi["raw_value"] = derived
        noi["normalized_value"] = derived
        noi["display_value"] = _format_display(derived, "USD", None)
        noi["status"] = "derived"
        noi["source_sheet"] = None
        noi["source_cell"] = formula  # shows the identity instead of a cell
        note = (f"Derived from pricing: {formula} = "
                f"{price:,.0f} × {cap:.4f} = {derived:,.0f}.")
        if prev is not None and abs(prev - derived) / max(derived, 1.0) > 0.10:
            note += (f" Extracted cell showed {prev:,.0f} — likely the wrong "
                     f"period/column; the pricing identity governs.")
        noi.setdefault("validation_notes", []).insert(0, note)


def _status_counts(records: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
