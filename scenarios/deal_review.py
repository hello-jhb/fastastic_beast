"""
Deal Review scenario — institutional acquisition memo (Path B architecture).

Architecture:
  - Catalog provides verified facts with cell-level provenance.
  - GPT acts as the analyst: reads catalog facts + raw file content +
    multi-year time series, then writes a deal memo.
  - Output adapts to deal type (ground-up dev, value-add, core, etc.)
    rather than forcing a rigid 30-field template.

Why this design:
  - Real analysts don't fill forms when they read closing files — they
    write a thesis. The output should match what an institutional asset
    manager actually produces.
  - Templates force every field to be populated, even when irrelevant
    (e.g. "Going-in NOI" on a ground-up dev = always $0). Adaptive
    sections handle that without "—" noise.
  - GPT is good at synthesis. Don't limit it to fill-in-the-blanks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ssot
from scenarios._llm import complete, llm_available
from scenarios.profiles import filter_layer_metrics
from flexible_extractor import extract_time_series_rows


UPLOAD_DIR = Path("uploads")


SYSTEM_PROMPT = """\
You are a senior real estate investment professional writing an acquisition memo.
The user is an institutional asset manager evaluating whether to fund this deal.

You have FOUR inputs:
  1. ANALYST CHECKLIST — BOUNDED METRICS (PRIMARY): the 25 metrics on an analyst's
     deal-review checklist, each carrying a status:
       VERIFIED        — single high-confidence candidate; safe to cite as fact.
       CANDIDATE_POOL  — multiple candidates passed validation; top-ranked taken.
                         Cite with cell reference; do NOT add editorial certainty.
       SUSPICIOUS      — failed schema validation. DO NOT cite as fact. Represent
                         in memo as "data quality issue" or omit field.
       MISSING         — no candidate found. Represent as "—" or omit field.
  2. LEGACY CATALOG FACTS — non-bounded metrics with cell references. Secondary
     source, only used for metrics not in the bounded checklist.
  3. PASS 2 INFERRED FIELDS — fields GPT inferred at ingest (property type, deal
     type, total debt). Note as "(inferred)" when cited.
  4. TIME SERIES — multi-year projections (NOI, revenue, expenses, cash flow)
     showing how the deal evolves. Use these for the NOI / cash flow trajectory.

CITATION RULES (NON-NEGOTIABLE):
- VERIFIED and CANDIDATE_POOL bounded metrics: cite with cell reference
  "$192M (General Information!C11)".
- SUSPICIOUS bounded metrics: NEVER cite the value. Note as "data quality issue"
  or omit the field entirely. The user is alerted via the audit-trail section.
- MISSING bounded metrics: write "—" or omit the field. Never invent.
- Inferred (Pass 2) fields: cite as "Multifamily (inferred)" — no cell ref.

YOUR JOB: write a SIMPLE deal memo. Four sections only: Snapshot,
Business Plan, Risks, Audit Appendix. The user will request deep dives
(capital structure, cash flow, returns, capex) via separate buttons — do NOT
include those sections in this output.

STYLE RULES:
- Snapshot: 3-5 sentences. Business Plan: EXACTLY 2 bullets. Risks: EXACTLY 2 bullets. Appendix: data only.
- Total output target: 250-400 words. Tight, executive-style.
- Use markdown.
- For Data Appendix metrics that are missing/suspicious, follow the explicit
  formatting rules below — do not silently omit them, the appendix is the audit.

STRUCTURE (write ONLY these four sections):

## Snapshot
One concise paragraph (3-5 sentences). Cover: asset name + property type (with
number of properties if portfolio), location, size (units / SF / keys),
acquisition date, purchase price, target hold, headline return (Levered IRR).
This is the elevator pitch. No filler.

## Business Plan
EXACTLY 2 sentences, bullet style. No paragraphs.
- Bullet 1: opportunity framing — what type of asset, what market, what makes
  it attractive in one sentence. Example: "Opportunity to invest in well-located
  Class-A multifamily in Glendale, CA submarket with strong rent fundamentals."
- Bullet 2: strategy + lever — Core / Core-Plus / Value-Add / Opportunistic /
  Ground-up Development, plus the specific value creation lever in one
  sentence. Example: "Value-Add play leveraging unit renovations and lease-up
  of vacant inventory to drive ~$2.5M NOI uplift."

## Risks
EXACTLY 2 bullets. Use verified facts, source-audited issues, and Pass 2
observations. Do not cite missing/suspicious values as facts.
- Bullet 1: execution or underwriting risk.
- Bullet 2: financing, exit, market, or data-quality risk.

## Audit Appendix
Render exactly these rows in this exact order. Use the DISPLAY LABEL on the
left (left of the arrow) as the line label — never write the word "Metric".
Pull the value from the bounded metric named on the right of the arrow.

Each line format:
  `**<Display Label>**: <display value> (<sheet>!<cell>)`   if verified/inferred/candidate_pool
  `**<Display Label>**: — (data quality issue)`             if suspicious
  `**<Display Label>**: —`                                  if missing

  Property Name        ← Asset Name
  Number of Properties ← Number of Properties
  Property Type        ← Property Type
  Location             ← Location
  Total SF             ← Total SF
  Total Units          ← Total Units
  Acquisition Date     ← Purchase Date
  Purchase Price       ← Purchase Price
  Total Project Cost   ← Total Project Cost
  Exit Date            ← Exit Date
  Exit Price           ← Exit Value / Terminal Value
  Hold Period          ← Hold Period
  Debt                 ← Debt Amount (acquisition/initial loan). If a
                         Construction Loan is also present, show BOTH on
                         separate lines: "Acquisition Loan: $X" and
                         "Construction Loan: $Y" — conversion/dev deals carry
                         both tranches (the construction loan funds the project
                         and repays the bridge).
  Leverage             ← Original LTV if present, ELSE Loan-to-Cost (LTC).
                         Label it "LTV: X%" or "LTC: X%" accordingly. Dev /
                         value-add deals are financed on cost (LTC); stabilized
                         acquisitions use LTV. Show whichever the model has; if
                         both, show both.
  Interest Rate        ← Interest Rate (see floating-rate rule below)
  Unlevered IRR        ← Unlevered IRR
  Levered IRR          ← Levered IRR
  Exit Cap Rate        ← Exit Cap Rate (if status is "not_applicable", show
                         "N/A" — dev/conversion deals have no going-in cap and
                         may express returns differently; never invent it)

CONFLICT RULE:
If a metric's status is "conflict" (two authoritative sources disagree), show
both with a ⚠ flag:
  `**<Label>**: ⚠ CONFLICT — $X (Sheet1!Cell1) vs $Y (Sheet2!Cell2)`
Never silently pick one.

FLOATING-RATE INTEREST DISPLAY RULE:
If Interest Rate Spread AND Interest Rate Cap are both populated, the debt is
FLOATING. Display Interest Rate as:
  `Interest Rate: <Spread>% spread + <Cap>% cap → <Spread+Cap>% max (Sheet!Cell)`
Otherwise display Interest Rate as a single rate.

DO NOT WRITE other sections in this output. Capital Structure, Cash Flow,
Return Profile, and CapEx Plan are on-demand deep-dives the user will request
via separate buttons. Do not generate them here.
"""


def _format_time_series_block(series: list[dict], max_rows: int = 25) -> str:
    """Render time series as a readable text table for GPT."""
    if not series:
        return "(no time series extracted from this file)"

    # Group by sheet, take most analytically relevant rows
    # Priority: NOI, Revenue, EGI, Operating Expenses, Cash Flow, Debt Service
    priority_terms = [
        "noi", "net operating income", "egi", "effective gross",
        "operating expense", "total expense", "cash flow",
        "debt service", "rental income", "total income",
        "potential gross", "occupancy", "stabilized",
        "total project", "total uses", "total sources", "equity funded",
    ]

    def row_priority(s):
        label_lower = s["label"].lower()
        for i, kw in enumerate(priority_terms):
            if kw in label_lower:
                return i
        return 999

    series_sorted = sorted(series, key=row_priority)[:max_rows]

    lines = []
    current_sheet = None
    for s in series_sorted:
        if s["sheet"] != current_sheet:
            current_sheet = s["sheet"]
            lines.append(f"\n[{current_sheet}]")
        headers = s.get("annual_headers") or s["headers"]
        values = s.get("annual_values") or s["values"]
        meta = ""
        if s.get("annualized"):
            meta = f" [annualized from monthly; {s.get('aggregation_method')}]"
        elif s.get("periodicity"):
            meta = f" [{s.get('periodicity')}]"

        # Format values
        vals = []
        for v in values[:8]:
            if v is None:
                vals.append("—")
            elif abs(v) >= 1_000_000:
                vals.append(f"${v/1_000_000:.2f}M")
            elif abs(v) >= 1_000:
                vals.append(f"${v/1_000:.0f}K")
            elif isinstance(v, float) and abs(v) < 1:
                vals.append(f"{v:.1%}")
            else:
                vals.append(f"{v:,.0f}")
        if headers:
            lines.append("  " + " | ".join(str(h) for h in headers[:8]))
        lines.append(f"  {(s['label'] + meta)[:70]:<70} {' | '.join(vals)}")
    return "\n".join(lines)


def _format_catalog_facts(metrics: dict) -> str:
    """Render catalog metrics as a citable list with cell references."""
    lines = []
    for name, data in metrics.items():
        if data.get("value") is None:
            continue
        val = data["value"]
        cell = f"{data.get('sheet','?')}!{data.get('cell','?')}"
        # Format number nicely
        if isinstance(val, (int, float)):
            if abs(val) >= 1_000_000:
                v_str = f"${val/1_000_000:.2f}M"
            elif abs(val) >= 1_000:
                v_str = f"${val:,.0f}"
            elif abs(val) < 1 and val != 0:
                v_str = f"{val:.2%}"
            else:
                v_str = f"{val:,.2f}"
        else:
            v_str = str(val)
        lines.append(f"  - **{name}**: {v_str}  ({cell})")
    return "\n".join(lines) if lines else "  (no catalog facts extracted)"


def _format_bounded_metrics(bounded: dict) -> str:
    """
    Render Phase 1 bounded metrics grouped by status, with cell provenance and
    explicit data-quality flags. This is the PRIMARY input to the memo —
    catalog-verified numbers with audit-grade citations.
    """
    if not bounded:
        return "(No bounded-metric extraction available — Phase 1 pipeline did not run.)"

    # Group by status
    verified, inferred, conflict, pool, suspicious, missing, na = [], [], [], [], [], [], []
    for name, rec in bounded.items():
        status = rec.get("status")
        if status == "verified":
            verified.append((name, rec))
        elif status in ("inferred", "derived"):
            inferred.append((name, rec))
        elif status == "conflict":
            conflict.append((name, rec))
        elif status == "candidate_pool":
            pool.append((name, rec))
        elif status == "suspicious":
            suspicious.append((name, rec))
        elif status == "not_applicable":
            na.append((name, rec))
        elif status == "missing":
            missing.append((name, rec))

    def _fmt_record(name, rec):
        val = rec["display_value"]
        sheet = rec.get("source_sheet")
        cell = rec.get("source_cell")
        cell_ref = f"{sheet}!{cell}" if sheet and cell else "—"
        period = rec.get("period")
        period_tag = f" [{period}]" if period and period != "n/a" else ""
        return f"  - **{name}**{period_tag}: {val}  ({cell_ref})"

    lines = []
    if verified:
        lines.append("VERIFIED (authoritative source, validated; safe to cite):")
        for name, rec in verified:
            lines.append(_fmt_record(name, rec))
        lines.append("")
    if inferred:
        lines.append("INFERRED (derived or GPT-inferred from verified context; cite, note as inferred):")
        for name, rec in inferred:
            lines.append(_fmt_record(name, rec))
        lines.append("")
    if conflict:
        lines.append("CONFLICT (authoritative sources DISAGREE — show both with ⚠, never pick one):")
        for name, rec in conflict:
            primary = _fmt_record(name, rec)
            confs = (rec.get("audit", {}) or {}).get("conflicts", [])
            alt_str = "; ".join(
                f"{c.get('sheet')}!{c.get('cell')}={c.get('value')}" for c in confs
            )
            lines.append(f"{primary}  ⚠ vs {alt_str}")
        lines.append("")
    if pool:
        lines.append("CANDIDATE POOL (multiple candidates passed schema; "
                     "top-ranked taken — cite with cell ref, no editorial certainty):")
        for name, rec in pool:
            lines.append(_fmt_record(name, rec))
        lines.append("")
    if suspicious:
        lines.append("SUSPICIOUS (failed schema validation — DO NOT cite as fact, "
                     "represent as data quality issue):")
        for name, rec in suspicious:
            notes = "; ".join(rec.get("validation_notes", []))[:160]
            lines.append(f"  - **{name}**: {rec['display_value']}  — {notes}")
        lines.append("")
    if na:
        lines.append("NOT APPLICABLE (legitimately N/A for this deal type — show as 'N/A', "
                     "do NOT treat as a data gap):")
        for name, rec in na:
            notes = "; ".join(rec.get("validation_notes", []))[:140]
            lines.append(f"  - **{name}**: N/A — {notes}")
        lines.append("")
    if missing:
        lines.append("MISSING (no candidates found in scanned sheets — represent as '—'):")
        for name, _ in missing:
            lines.append(f"  - **{name}**")
        lines.append("")

    return "\n".join(lines).strip()


def _format_pass2_fields(raw_insights: dict) -> str:
    """Render Pass 2 found fields as a list."""
    if not raw_insights:
        return "(Pass 2 did not run — no inferred fields available)"
    found = raw_insights.get("found", {}) or {}
    if not found:
        return "(Pass 2 ran but found no additional fields)"
    lines = []
    for field_name, data in found.items():
        if not isinstance(data, dict) or data.get("value") is None:
            continue
        val = data["value"]
        label = data.get("label_in_file", "")
        sheet = data.get("sheet", "")
        loc = f" [{sheet}: {label}]" if label or sheet else ""
        lines.append(f"  - **{field_name}**: {val}{loc}")
    return "\n".join(lines) if lines else "(no fields populated)"


def generate_deal_review() -> dict[str, Any]:
    """
    Generate the institutional deal memo.
    """
    s = ssot.load_ssot()
    underwriting = s["layers"].get("underwriting")
    if not underwriting:
        return {"error": "No underwriting layer in SSOT. Upload an acquisition file first."}

    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    # Apply scenario profile so we only pass relevant catalog metrics
    filtered = filter_layer_metrics(underwriting, "deal_review")
    catalog_metrics = filtered.get("metrics", {})

    # Pass 2 inferred fields
    raw_insights = underwriting.get("raw_insights") or {}

    # Time series from the source file (NOI/revenue/cash flow trajectory)
    source_file = underwriting.get("source_file")
    time_series_block = ""
    if source_file:
        file_path = UPLOAD_DIR / source_file
        if file_path.exists():
            try:
                ts = extract_time_series_rows(file_path)
                time_series_block = _format_time_series_block(ts)
            except Exception as e:
                time_series_block = f"(time series extraction failed: {e})"
        else:
            time_series_block = f"(source file not found in uploads: {source_file})"

    # Pass 2 observations (free-form context GPT noted at ingest)
    observations = raw_insights.get("observations", []) or []
    model_summary = raw_insights.get("model_summary", "") or ""

    # Phase 1 — bounded analyst-checklist metrics with schema validation
    bounded_metrics = underwriting.get("bounded_metrics", {}) or {}

    # Build the user prompt
    user_prompt = f"""\
ASSET: {source_file or 'Unknown'}
INGESTED: {underwriting.get('ingested_at', 'Unknown')}

{f'PASS 2 MODEL SUMMARY: {model_summary}' if model_summary else ''}

===== ANALYST CHECKLIST — BOUNDED METRICS (PRIMARY SOURCE) =====
These 25 metrics are the analyst's deal-review checklist, each schema-validated
with explicit provenance. Status-based citation rules apply (see system prompt).

{_format_bounded_metrics(bounded_metrics)}

===== LEGACY CATALOG FACTS (secondary — use only for metrics not in the bounded list above) =====

{_format_catalog_facts(catalog_metrics)}

===== PASS 2 INFERRED FIELDS (use; note as "(inferred)" if cited) =====

{_format_pass2_fields(raw_insights)}

===== PASS 2 OBSERVATIONS (use for context / risks) =====

{chr(10).join(f'  - {o}' for o in observations) if observations else '  (none)'}

===== TIME SERIES (multi-year projections — use for NOI / cash flow trajectory) =====
{time_series_block}

Now write the deal memo following the structure in your system prompt.
Adapt sections to this deal's type. Be specific. Cite cell references where
catalog facts are used. NEVER cite SUSPICIOUS or MISSING values as fact.
"""

    narrative = complete(SYSTEM_PROMPT, user_prompt, temperature=0.2)

    # Memorialize the acquisition (write-once)
    _memorialize_acquisition(s, narrative, filtered, underwriting)

    # Bounded-metric status breakdown for diagnostics
    bounded_status_counts: dict[str, int] = {}
    for rec in bounded_metrics.values():
        s = rec.get("status", "unknown")
        bounded_status_counts[s] = bounded_status_counts.get(s, 0) + 1

    return {
        "scenario": "deal_review",
        "narrative": narrative,
        "data_used": {
            "layers": ["underwriting"],
            "source_files": [source_file] if source_file else [],
            "bounded_metric_count": len(bounded_metrics),
            "bounded_status_counts": bounded_status_counts,
            "catalog_metric_count": len(catalog_metrics),
            "pass2_field_count": len(raw_insights.get("found", {}) if raw_insights else {}),
            "time_series_rows": len(time_series_block.splitlines()) if time_series_block else 0,
        },
    }


def _memorialize_acquisition(
    s: dict[str, Any],
    narrative: str,
    filtered: dict[str, Any],
    underwriting: dict[str, Any],
) -> None:
    """Save the acquisition memo as a permanent record (write-once)."""
    if s["layers"].get("acquisition_summary"):
        return
    s["layers"]["acquisition_summary"] = {
        "source_file": underwriting.get("source_file"),
        "ingested_at": underwriting.get("ingested_at"),
        "metric_count": filtered.get("metric_count", 0),
        "metrics":      filtered.get("metrics", {}),
        "narrative":    narrative,
    }
    ssot.save_ssot(s)
