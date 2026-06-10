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

def _bounded_pretty(bounded: dict, metric_names: list[str]) -> str:
    """Pretty-print a subset of bounded metrics for a deep-dive prompt."""
    if not bounded:
        return "(no bounded metrics extracted)"
    lines = []
    for name in metric_names:
        rec = bounded.get(name)
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

Output format (markdown):

## Capital Structure

| Component | Amount | Source |
|---|---|---|
| Purchase Price | $X (Sheet!Cell) | ... |
| Total Project Cost | $X | ... |
| Acquisition Loan | $X | ... |
| Construction Loan | $X (only if present — conversion/dev) | ... |
| Equity Required | $X | ... |
| LTV or LTC | X% | LTC for cost-financed dev/value-add |
| Interest Rate | (see floating rule above) | ... |
| Loan Maturity | X months | ... |
| I/O Period | X months | ... |
| DSCR | X.Xx | ... |
| Debt Yield | X.X% | ... |

Omit rows that are missing/N/A rather than showing "—" clutter. If both an
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
    user_prompt = (
        "Bounded metrics for Capital Structure:\n\n"
        + _bounded_pretty(bounded, relevant)
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

Output format (markdown):

## Cash Flow / NOI Trajectory

| Period | NOI | Note |
|---|---|---|
| Year 1 (going-in) | $X (Sheet!Cell) | ... |
| Stabilized | $X | ... |
| Exit | $X | ... |

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

    # Time series from the source workbook
    source_file = uw.get("source_file")
    ts_block = ""
    if source_file:
        fp = UPLOAD_DIR / source_file
        if fp.exists():
            try:
                ts = extract_time_series_rows(fp)
                # Only the NOI/Revenue/Expense series
                relevant_ts = [
                    r for r in ts
                    if any(k in r["label"].lower() for k in (
                        "noi", "net operating income", "revenue",
                        "egi", "gross income", "operating expense", "cash flow",
                    ))
                ][:20]
                if relevant_ts:
                    ts_lines = ["", "Time series (relevant rows):"]
                    for s in relevant_ts:
                        values = s.get("annual_values") or s["values"]
                        headers = s.get("annual_headers") or s.get("headers") or []
                        meta = ""
                        if s.get("annualized"):
                            meta = f" [annualized from monthly; {s.get('aggregation_method')}]"
                        elif s.get("periodicity"):
                            meta = f" [{s.get('periodicity')}]"
                        vals = " | ".join(
                            f"{v:,.0f}" if isinstance(v, (int, float)) and v else "—"
                            for v in values[:8]
                        )
                        header_str = " | ".join(str(h) for h in headers[:8])
                        ts_lines.append(f"  [{s['sheet']}] {s['label']}{meta}: {header_str} => {vals}")
                    ts_block = "\n".join(ts_lines)
            except Exception:
                pass

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

Output format (markdown):

## Return Profile

| Metric | Value | Source |
|---|---|---|
| Levered IRR | X% | (Sheet!Cell) |
| Unlevered IRR | X% | ... |
| Equity Multiple | X.Xx | ... |
| Going-In Cap Rate | X.X% | ... |
| Exit Cap Rate | X.X% | ... |
| Exit Value | $X | ... |
| Hold Period | X years | ... |

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

Output format (markdown):

## CapEx Plan

| Item | Amount | Source |
|---|---|---|
| Total CapEx Budget | $X (Sheet!Cell) | ... |
| Total Project Cost | $X | ... |
| Hold Period | X years | ... |

If a multi-year draw schedule is in the time series, render it as a small table:

| Year | CapEx Draw |
|---|---|
| Year 1 | $X |
| Year 2 | $X |

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
            try:
                ts = extract_time_series_rows(fp)
                relevant_ts = [
                    r for r in ts
                    if any(k in r["label"].lower() for k in (
                        "capex", "hard cost", "soft cost", "construction",
                        "total project", "draw", "tenant improvement", "ti",
                    ))
                ][:25]
                if relevant_ts:
                    ts_lines = ["", "CapEx-related time series:"]
                    for s in relevant_ts:
                        vals = " | ".join(
                            f"{v:,.0f}" if isinstance(v, (int, float)) and v else "—"
                            for v in s["values"][:8]
                        )
                        ts_lines.append(f"  [{s['sheet']}] {s['label']}: {vals}")
                    ts_block = "\n".join(ts_lines)
            except Exception:
                pass

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
