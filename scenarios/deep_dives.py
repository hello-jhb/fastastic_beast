"""
deep_dives.py — focused deep-dive sections for the Deal Review workspace.

Each function generates a single thematic section the user can request
on-demand via UI buttons:

    - Capital Structure
    - Cash Flow / NOI Trajectory
    - Return Profile
    - CapEx Plan
    - Key Risks

Each deep dive reads from the same verified SSOT (bounded_metrics, raw_insights,
time series) the main memo uses, but with a tighter prompt scoped to its topic.
This keeps the main memo short (snapshot + thesis + appendix) while letting
the user drill into any aspect with one click.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import ssot
from scenarios._llm import complete, llm_available
from flexible_extractor import extract_time_series_rows


UPLOAD_DIR = Path("uploads")


# ---------------------------------------------------------------------------
# Shared context builders
# ---------------------------------------------------------------------------

_NAME_TO_ID: dict[str, str] | None = None


def _name_to_id() -> dict[str, str]:
    """Lazy normalized {name-or-alias: metric_id} index over the catalog."""
    global _NAME_TO_ID
    if _NAME_TO_ID is None:
        from metric_catalog import load_metric_catalog
        from flexible_extractor import normalize_text
        idx: dict[str, str] = {}
        for m in load_metric_catalog():
            idx[normalize_text(m["metric_name"])] = m["metric_id"]
            for a in m.get("aliases", []) or []:
                idx.setdefault(normalize_text(a), m["metric_id"])
        _NAME_TO_ID = idx
    return _NAME_TO_ID


def _resolve_bounded(bounded: dict, name: str) -> dict | None:
    """
    Look up a bounded record by canonical name, falling back to an alias/id
    match so a key mismatch can never silently drop a metric (e.g. Levered IRR
    stored under a non-canonical key).
    """
    rec = bounded.get(name)
    if rec:
        return rec
    from flexible_extractor import normalize_text
    mid = _name_to_id().get(normalize_text(name))
    if mid:
        for r in bounded.values():
            if r.get("metric_id") == mid:
                return r
    return None


def _ts_block(file_path, keywords: tuple[str, ...], max_rows: int = 16,
              header: str = "Time series (relevant rows):") -> str:
    """
    Parser-backed time-series block filtered to rows whose label matches any
    keyword. Authoritative periodicity + annualization come from the model
    parser; falls back to the heuristic extractor when no tables are found.
    """
    try:
        from financial_model_parser import build_time_series
        ts = build_time_series(file_path) or extract_time_series_rows(file_path)
    except Exception:
        return ""
    rel = [r for r in ts if any(k in r["label"].lower() for k in keywords)][:max_rows]
    if not rel:
        return ""
    lines = ["", header]
    for s in rel:
        values = s.get("annual_values") or s.get("values") or []
        headers = s.get("annual_headers") or s.get("headers") or []
        if s.get("annualized"):
            meta = f" [annualized from {s.get('periodicity')}; {s.get('aggregation_method')}]"
        elif s.get("periodicity"):
            meta = f" [{s.get('periodicity')}]"
        else:
            meta = ""
        vals = " | ".join(
            f"{v:,.0f}" if isinstance(v, (int, float)) and v else "—"
            for v in values[:8]
        )
        header_str = " | ".join(str(h) for h in headers[:8])
        lines.append(f"  [{s['sheet']}] {s['label']}{meta}: {header_str} => {vals}")
    return "\n".join(lines)


def _bounded_pretty(bounded: dict, metric_names: list[str]) -> str:
    """Pretty-print a subset of bounded metrics for a deep-dive prompt."""
    if not bounded:
        return "(no bounded metrics extracted)"
    lines = []
    for name in metric_names:
        rec = _resolve_bounded(bounded, name)
        if not rec:
            lines.append(f"  - {name}: MISSING")
            continue
        status = rec.get("status")
        val = rec.get("display_value", "—")
        sheet = rec.get("source_sheet")
        cell = rec.get("source_cell")
        cell_ref = f"{sheet}!{cell}" if sheet and cell else "—"
        if status in ("verified", "candidate_pool"):
            lines.append(f"  - **{name}**: {val} ({cell_ref})")
        elif status == "suspicious":
            notes = "; ".join((rec.get("validation_notes") or []))[:100]
            lines.append(f"  - **{name}**: SUSPICIOUS — {notes}")
        else:
            lines.append(f"  - **{name}**: —")
    return "\n".join(lines)


def _load_uw_layer() -> dict[str, Any] | None:
    s = ssot.load_ssot()
    return s["layers"].get("underwriting")


# ---------------------------------------------------------------------------
# Capital Structure
# ---------------------------------------------------------------------------
_CAPITAL_STRUCTURE_SYSTEM = """\
You are writing the Capital Structure section of a real estate IC memo.
Use ONLY the provided metrics with cell references. Cite cell references.
For floating-rate debt (when Interest Rate Spread + Cap are both present),
explain the floating structure: spread, cap strike, max effective rate.

Output format (markdown). Use BULLET POINTS for all figures — NEVER markdown
tables (they are hard to read in this app):

## Capital Structure

- **Purchase Price:** $X (Sheet!Cell)
- **Total Project Cost:** $X (Sheet!Cell)
- **Acquisition Loan:** $X (Sheet!Cell)
- **Construction Loan:** $X — only if present (conversion/dev)
- **Equity Required:** $X
- **LTV or LTC:** X% — LTC for cost-financed dev/value-add
- **Interest Rate:** (see floating rule above)
- **Loan Maturity:** X months
- **I/O Period:** X months
- **DSCR:** X.Xx
- **Debt Yield:** X.X%

Omit bullets that are missing/N/A rather than showing "—" clutter. If both an
acquisition loan and a construction loan are present, note that the
construction loan funds the project and typically repays the acquisition
bridge. Then 1-2 sentences on the capital stack's risk/return profile.
Max 200 words total. No filler.
"""


def deep_dive_capital_structure() -> dict[str, Any]:
    uw = _load_uw_layer()
    if not uw:
        return {"error": "No underwriting layer in SSOT."}
    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    bounded = uw.get("bounded_metrics", {}) or {}
    relevant = [
        "Purchase Price", "Total Project Cost", "Debt Amount", "Construction Loan",
        "Equity Invested",
        "Original LTV", "Loan-to-Cost (LTC)", "Interest Rate", "Interest Rate Spread", "Interest Rate Cap",
        "Loan Maturity", "Interest-Only Period Remaining",
        "DSCR / Debt Coverage Ratio", "Debt Yield",
    ]
    # Structured debt schedule (amort / interest / balance over time), parsed.
    debt_block = ""
    source_file = uw.get("source_file")
    if source_file:
        fp = UPLOAD_DIR / source_file
        if fp.exists():
            debt_block = _ts_block(
                fp,
                ("debt service", "interest expense", "interest", "amortization",
                 "principal", "loan balance", "debt balance", "maturity", "debt addition"),
                header="Debt schedule (parsed from debt / cash-flow tables — periodicity-aware):",
            )

    user_prompt = (
        "Bounded metrics for Capital Structure:\n\n"
        + _bounded_pretty(bounded, relevant)
        + debt_block
        + "\n\nWrite the Capital Structure section."
    )
    text = complete(_CAPITAL_STRUCTURE_SYSTEM, user_prompt, temperature=0.1)
    return {"section": "capital_structure", "narrative": text}


# ---------------------------------------------------------------------------
# Cash Flow / NOI Trajectory
# ---------------------------------------------------------------------------
_CASH_FLOW_SYSTEM = """\
You are writing the Cash Flow / NOI Trajectory section of a real estate IC memo.
Use the bounded metrics AND the time-series data provided. Cite cell references.

Walk through how NOI evolves:
  - Year 1 (going-in) NOI level
  - Stabilization year and stabilized NOI
  - Exit NOI
  - Identify the trajectory shape (flat/growth/dev ramp-up/value-add lift)

Output format (markdown). Use BULLET POINTS for all figures — NEVER markdown
tables (they are hard to read in this app):

## Cash Flow / NOI Trajectory

- **Year 1 (going-in) NOI:** $X (Sheet!Cell)
- **Stabilized NOI:** $X — stabilization year
- **Exit NOI:** $X

Then 2-3 sentences explaining the trajectory and the drivers.
Max 250 words total.
"""


def deep_dive_cash_flow() -> dict[str, Any]:
    uw = _load_uw_layer()
    if not uw:
        return {"error": "No underwriting layer in SSOT."}
    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    bounded = uw.get("bounded_metrics", {}) or {}
    relevant = [
        "Net Operating Income (NOI)", "Exit NOI",
        "Going-in Cap Rate", "Exit Cap Rate",
        "Exit Value / Terminal Value", "Hold Period",
        "Total Units", "Total SF",
    ]

    # Time series from the source workbook (parser-backed, periodicity-aware).
    source_file = uw.get("source_file")
    ts_block = ""
    if source_file:
        fp = UPLOAD_DIR / source_file
        if fp.exists():
            ts_block = _ts_block(fp, (
                "noi", "net operating income", "revenue",
                "egi", "gross income", "operating expense", "cash flow",
            ), max_rows=20)

    user_prompt = (
        "Bounded metrics for Cash Flow / NOI:\n\n"
        + _bounded_pretty(bounded, relevant)
        + ts_block
        + "\n\nWrite the Cash Flow / NOI Trajectory section."
    )
    text = complete(_CASH_FLOW_SYSTEM, user_prompt, temperature=0.1)
    return {"section": "cash_flow", "narrative": text}


# ---------------------------------------------------------------------------
# Return Profile
# ---------------------------------------------------------------------------
_RETURN_PROFILE_SYSTEM = """\
You are writing the Return Profile section of a real estate IC memo.
Use ONLY bounded metrics with cell references.

Output format (markdown). Use BULLET POINTS for all figures — NEVER markdown
tables (they are hard to read in this app):

## Return Profile

- **Levered IRR:** X% (Sheet!Cell)
- **Unlevered IRR:** X%
- **Equity Multiple:** X.Xx
- **Going-In Cap Rate:** X.X%
- **Exit Cap Rate:** X.X%
- **Exit Value:** $X
- **Hold Period:** X years

Then 2-3 sentences explaining where the return is coming from
(yield/cap compression/operational uplift/development premium) — based on
the cap rate spread, NOI trajectory, and hold period.
Max 200 words total.
"""


def deep_dive_return_profile() -> dict[str, Any]:
    uw = _load_uw_layer()
    if not uw:
        return {"error": "No underwriting layer in SSOT."}
    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    bounded = uw.get("bounded_metrics", {}) or {}
    relevant = [
        "Levered IRR", "Unlevered IRR", "Equity Multiple",
        "Going-in Cap Rate", "Exit Cap Rate", "Exit Value / Terminal Value",
        "Hold Period", "Net Operating Income (NOI)", "Exit NOI",
    ]
    user_prompt = (
        "Bounded metrics for Return Profile:\n\n"
        + _bounded_pretty(bounded, relevant)
        + "\n\nWrite the Return Profile section."
    )
    text = complete(_RETURN_PROFILE_SYSTEM, user_prompt, temperature=0.1)
    return {"section": "return_profile", "narrative": text}


# ---------------------------------------------------------------------------
# CapEx Plan
# ---------------------------------------------------------------------------
_CAPEX_PLAN_SYSTEM = """\
You are writing the CapEx Plan section of a real estate IC memo.
Use the bounded metrics AND any time-series data provided. Cite cell references.

Output format (markdown). Use BULLET POINTS for all figures — NEVER markdown
tables (they are hard to read in this app):

## CapEx Plan

- **Total CapEx Budget:** $X (Sheet!Cell)
- **Total Project Cost:** $X
- **Hold Period:** X years

If a multi-year draw schedule is in the time series, list it as bullets:

- **Year 1:** $X
- **Year 2:** $X

Then 2-3 sentences explaining the CapEx allocation (deferred maintenance,
unit renovation, building systems, ground-up construction, etc.).
Max 250 words total.
"""


def deep_dive_capex_plan() -> dict[str, Any]:
    uw = _load_uw_layer()
    if not uw:
        return {"error": "No underwriting layer in SSOT."}
    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    bounded = uw.get("bounded_metrics", {}) or {}
    relevant = [
        "CapEx Budget", "Total Project Cost", "Purchase Price",
        "Hold Period", "Total Units", "Total SF",
    ]

    source_file = uw.get("source_file")
    ts_block = ""
    if source_file:
        fp = UPLOAD_DIR / source_file
        if fp.exists():
            ts_block = _ts_block(fp, (
                "capex", "hard cost", "soft cost", "construction",
                "total project", "draw", "tenant improvement", "ti",
            ), max_rows=25, header="CapEx-related time series (periodicity-aware):")

    user_prompt = (
        "Bounded metrics for CapEx Plan:\n\n"
        + _bounded_pretty(bounded, relevant)
        + ts_block
        + "\n\nWrite the CapEx Plan section."
    )
    text = complete(_CAPEX_PLAN_SYSTEM, user_prompt, temperature=0.1)
    return {"section": "capex_plan", "narrative": text}


# ---------------------------------------------------------------------------
# Key Risks
# ---------------------------------------------------------------------------
_KEY_RISKS_SYSTEM = """\
You are writing the Key Risks section of a real estate IC memo.
Identify 3-5 risks that are SPECIFIC TO THIS DEAL based on the metrics and
context provided. Each risk must reference a specific number, cell, or
inferred characteristic.

FORBIDDEN: generic boilerplate risks ("market risk", "interest rate risk")
without a model-grounded basis. If you cite "interest rate risk," it must
tie to a specific assumption in the model (floating rate exposure, refinance
risk at a specific maturity, etc.).

Output format (markdown):

## Key Risks

1. **Risk Title** — One sentence with the specific data point that creates
   the risk. Cite cell reference where possible.

2. **Risk Title** — ...

3. **Risk Title** — ...

(3-5 items total. Max 250 words.)
"""


def deep_dive_key_risks() -> dict[str, Any]:
    uw = _load_uw_layer()
    if not uw:
        return {"error": "No underwriting layer in SSOT."}
    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    bounded = uw.get("bounded_metrics", {}) or {}
    relevant = list(bounded.keys())  # all bounded metrics — risks may come from anywhere

    raw_insights = uw.get("raw_insights") or {}
    observations = raw_insights.get("observations", []) or []
    model_summary = raw_insights.get("model_summary", "") or ""

    user_prompt = (
        f"Model summary: {model_summary}\n\n"
        "All bounded metrics:\n\n"
        + _bounded_pretty(bounded, relevant)
        + "\n\nAdditional context from Pass 2 observations:\n"
        + ("\n".join(f"  - {o}" for o in observations) if observations else "  (none)")
        + "\n\nWrite the Key Risks section."
    )
    text = complete(_KEY_RISKS_SYSTEM, user_prompt, temperature=0.2)
    return {"section": "key_risks", "narrative": text}


# ---------------------------------------------------------------------------
# Dispatcher (used by app + tools)
# ---------------------------------------------------------------------------

DEEP_DIVES: dict[str, Any] = {
    "capital_structure": deep_dive_capital_structure,
    "cash_flow":         deep_dive_cash_flow,
    "return_profile":    deep_dive_return_profile,
    "capex_plan":        deep_dive_capex_plan,
    "key_risks":         deep_dive_key_risks,
}


def run_deep_dive(name: str) -> dict[str, Any]:
    fn = DEEP_DIVES.get(name)
    if not fn:
        return {"error": f"Unknown deep dive: {name}. Valid: {sorted(DEEP_DIVES.keys())}"}
    return fn()
