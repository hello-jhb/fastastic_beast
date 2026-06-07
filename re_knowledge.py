"""
re_knowledge.py — centralized real estate domain knowledge.

Single source of truth for the conventions, definitions, and semantics that
every GPT call in the system relies on. Previously this knowledge was scattered
across prompt strings in metric_resolver_gpt.py, metric_fallback.py, _llm.py,
and deal_review.py — inconsistent and unversioned.

Now every GPT-touching module imports its prompt fragments from here. To tune
how the system reasons about real estate, edit THIS file. Bump KNOWLEDGE_VERSION
when a change should invalidate cached extractions.

Design principle: this is *bounded adaptation*. GPT operates within these
documented conventions (so it's consistent and predictable) but can still
handle files and labels it has never seen (so it's adaptive).
"""
from __future__ import annotations

KNOWLEDGE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Period semantics — the #1 source of wrong picks is period confusion
# ---------------------------------------------------------------------------
PERIOD_GLOSSARY = """\
PERIOD GLOSSARY (period confusion is the most common extraction error):
  at_close   = value at acquisition / first day of ownership.
               Labels: "100% Purchase Price", "Going-In LTV", "Closing Equity",
               "At Close", "Day 1".
  year_1     = first projection year, a.k.a. "Going-In" or "T-12 / Trailing".
               Labels: "Going-In NOI", "Year 1 NOI", "NOI to Determine
               Going-In Cap Rate". This is NOT the stabilized figure.
  stabilized = post-business-plan / post-lease-up steady state.
               Labels: "Stabilized NOI", "Stabilized Yield", "Stabilized Cap".
  exit       = at disposition / sale.
               Labels: "Exit NOI", "Sale Price", "Terminal Cap Rate",
               "Reversion", "Year of Sale".
  n/a        = period doesn't apply (counts, whole-deal ratios)."""


# ---------------------------------------------------------------------------
# Deal-level vs unit-level conventions
# ---------------------------------------------------------------------------
DEAL_LEVEL_CONVENTIONS = """\
DEAL-LEVEL vs UNIT-LEVEL:
  - Values labeled "100%", "Total", "Aggregate", "All-in", "Combined",
    "Consolidated", "Portfolio" are DEAL-LEVEL. Prefer these.
  - REJECT "per Unit", "per Key", "per SF", "per Door", "per Property",
    or single-property breakdowns when a deal-level total is requested.
  - A grand-total row beats any sub-category (hard costs only, soft costs
    only, contingency, F&B, etc.)."""


# ---------------------------------------------------------------------------
# Debt conventions (floating-rate structure is frequently mis-extracted)
# ---------------------------------------------------------------------------
DEBT_CONVENTIONS = """\
DEBT CONVENTIONS:
  - Floating-rate debt has three parts: Spread + Index (LIBOR/SOFR) + optional
    Cap. "Interest Rate" usually means the all-in rate (Spread + Index). If a
    separate "Interest Rate Spread" exists, then a bare "Spread" cell is the
    spread component only — do not report it as the full interest rate.
  - Max effective floating rate ≈ Spread + Cap strike.
  - LTV is debt as a % of value (or cost). LTC is debt as a % of total cost.
  - DSCR = NOI / annual debt service. Debt Yield = NOI / loan amount."""


# ---------------------------------------------------------------------------
# Property-type conventions — what each asset class actually tracks
# ---------------------------------------------------------------------------
PROPERTY_TYPE_CONVENTIONS = """\
PROPERTY-TYPE CONVENTIONS (what each asset class measures):
  Hotel        — counts KEYS/ROOMS (not SF). Tracks ADR, RevPAR, Occupancy.
  Multifamily  — counts UNITS/DOORS. Tracks rent/unit, occupancy.
  Office       — measures rentable SF (RSF/NRA). Tracks rent/SF, WALT.
  Industrial   — measures SF. Tracks rent/SF, clear height.
  Retail       — measures GLA (SF). Tracks rent/SF, sales/SF, anchor mix.
  Senior Living— counts beds/units. Tracks care level mix.
  Self-Storage — counts units + SF.
  A deal's property type is often encoded as a breakdown like
  "Asset Type - Hotel % = 100%" rather than a literal "Hotel" cell — infer
  the type from whichever asset-type-% row equals (or is closest to) 100%."""


# ---------------------------------------------------------------------------
# Identity relationships — used both for GPT context and deterministic checks
# ---------------------------------------------------------------------------
IDENTITY_RELATIONSHIPS = """\
ARITHMETIC IDENTITIES (a correct extraction satisfies these):
  - Going-In Cap Rate ≈ Year-1 NOI / Purchase Price
  - LTV ≈ Debt / Purchase Price  (or Debt / Total Cost for LTC)
  - Sources = Uses: Debt + Equity ≈ Total Project Cost
  - Exit Value ≈ Exit NOI / Exit Cap Rate
  - Equity Multiple is directionally consistent with IRR over the hold period
  If a candidate value breaks one of these by >5-10%, it is suspect."""


# ---------------------------------------------------------------------------
# Sheet-role vocabulary — what kinds of sheets a model contains
# ---------------------------------------------------------------------------
SHEET_ROLE_VOCAB = """\
SHEET ROLES (classify each sheet by what it CONTAINS, not its name):
  summary       — deal-level summary / one-pager / key UW metrics / general info
  inputs        — assumptions / inputs tab driving the model (rent, growth,
                  unit mix, purchase price, debt terms entered as assumptions)
  cash_flow     — multi-year operating proforma (revenue, expenses, NOI by year)
  sources_uses  — sources & uses / capitalization table at close
  debt          — loan terms, covenants, amortization, debt schedule
  capex         — capital expenditure budget / draw schedule / hard+soft costs
  returns       — IRR / equity multiple / waterfall / return sensitivity
  rent_roll     — tenant-by-tenant or unit-by-unit rent detail
  comps         — sales / rent / market comparables (NOT the subject deal)
  sensitivity   — scenario / sensitivity tables (alternative assumptions)
  backup        — supporting detail / source data that feeds primary sheets
  market        — macro / submarket / supply-demand data
  other         — navigation tabs, blanks, anything uncategorized"""


# Map each sheet role to the extraction priority tier it implies.
# Used when GPT classification overrides the name-based tier guess.
ROLE_TO_TIER: dict[str, int] = {
    "summary":      1,
    "inputs":       1,   # assumptions/inputs tab is authoritative for deal basis
    "sources_uses": 1,   # sources & uses is authoritative for basis/debt/equity
    "cash_flow":    2,
    "capex":        3,
    "debt":         4,
    "returns":      5,
    "rent_roll":    6,
    "market":       6,
    "other":        6,
    # Roles we intentionally skip during bulk extraction (would pollute metrics)
    "comps":        99,
    "sensitivity":  99,
    "backup":       99,
}

# Canonical role list (for validation of catalog source-hierarchy entries)
ALL_ROLES = [
    "summary", "inputs", "sources_uses", "cash_flow", "capex", "debt",
    "returns", "rent_roll", "comps", "sensitivity", "backup", "market", "other",
]


# ---------------------------------------------------------------------------
# Assembled prompt blocks
# ---------------------------------------------------------------------------

def knowledge_block(*, include: list[str] | None = None) -> str:
    """
    Return a combined knowledge block for injection into a GPT system prompt.

    include: optional list of section keys to include. If None, include all
    the "extraction" sections (period, deal-level, debt, property-type,
    identities). Sheet-role vocab is requested explicitly by the classifier.

    Available keys: period, deal_level, debt, property_type, identities,
                    sheet_roles
    """
    sections = {
        "period":        PERIOD_GLOSSARY,
        "deal_level":    DEAL_LEVEL_CONVENTIONS,
        "debt":          DEBT_CONVENTIONS,
        "property_type": PROPERTY_TYPE_CONVENTIONS,
        "identities":    IDENTITY_RELATIONSHIPS,
        "sheet_roles":   SHEET_ROLE_VOCAB,
    }
    if include is None:
        include = ["period", "deal_level", "debt", "property_type", "identities"]
    return "\n\n".join(sections[k] for k in include if k in sections)
