"""
Deal Review scenario.

Purpose: Memorialize the acquisition event and establish the SSOT baseline.
         This is the "founding document" of an asset — it records what was
         underwritten at purchase so everything after (actuals, revisions)
         can be compared against it.

Input:   Underwriting model (Excel). IC memo, closing docs (future: PDF).
Output:  Structured acquisition summary in a fixed template format.

Hard constraints:
- Output follows the template EXACTLY — no prose outside designated fields.
- Every number must come from the SSOT. Missing values show as "—".
- Strategy classification may be inferred from deal characteristics.
- Risk/Mitigant: 2 bullets only unless user asks for more.
"""

from __future__ import annotations

import json
from typing import Any

import ssot
from scenarios._llm import complete, llm_available
from scenarios.profiles import filter_layer_metrics


SYSTEM_PROMPT = """\
You are an institutional real estate asset manager writing a formal acquisition summary.

Your job is to populate a structured deal template using ONLY the metrics provided.
This document memorializes the original investment thesis at the time of acquisition.

HARD RULES:
1. Output ONLY the template structure below. Do not add sections, prose, or commentary outside the defined fields.
2. Every dollar amount and percentage must come from the provided metrics. If a value is not available, write "—".
3. Do NOT calculate or derive values not explicitly present (exception: simple ratios if both inputs are provided).
4. Strategy (Opportunistic / Value-Add / Core / Core-Plus) MAY be inferred from deal characteristics if not explicit:
   - Core:       stabilized, low vacancy, institutional market, sub-6% going-in cap
   - Core-Plus:  mostly stabilized, minor lease-up, 6-7% cap
   - Value-Add:  significant vacancy, renovation, lease-up required, 7%+ cap or below-market rents
   - Opportunistic: distressed, development, major repositioning, high execution risk
5. Risk/Mitigant: write exactly 2 bullets unless the user explicitly asks for more.
6. Format all dollar values as $X,XXX,XXX. Format percentages as X.X%. Format multiples as X.Xx.
7. If the same metric appears in multiple categories, use the most specific value.
"""


TEMPLATE = """\
Populate this acquisition summary using the metrics below. Replace every [bracket] with the actual value or "—" if not available.

METRICS FROM UNDERWRITING MODEL:
{metrics_json}

---

OUTPUT THIS TEMPLATE EXACTLY:

## [Asset Name if known, otherwise: Acquisition Summary]

### Building Information
| | |
|---|---|
| Property Type | [type — infer from context if not explicit] |
| Total SF / Units | [sf or unit count] |
| Current Occupancy at Purchase | [% from T12 or UW assumption] |
| T12 / Going-in NOI | $[amount] |
| NOI Margin | [%] |

---

### Deal Summary
| | |
|---|---|
| Purchase Price | $[amount] |
| Strategy | [Opportunistic / Value-Add / Core-Plus / Core] |
| Strategy Description | [One sentence: what is the play?] |
| Capital Outlay After Closing | $[CapEx / TI / LC budget] |
| Total All-in Basis | $[amount] |
| Hold Period | [X years] |

---

### Debt & Equity
| | |
|---|---|
| Initial Debt Funding | $[amount] |
| Future Funding (CapEx / TI / LC draws) | $[amount] |
| Total Debt | $[amount] |
| Term | [X months I/O + X months amortizing, or as stated] |
| Interest Rate | [X.X%] |
| LTV | [X.X%] |
| LTC | [X.X%] |
| Underwritten DSCR | [X.Xx] |
| Underwritten Debt Yield | [X.X%] |
| Break-even Occupancy | [X.X%] |
| Equity Invested | $[amount] |
| LP / GP Split | [XX% LP / XX% GP — or "—" if not in model] |

---

### NOI Projection
| | |
|---|---|
| Going-in NOI (at purchase) | $[amount] |
| Stabilized NOI (target) | $[amount] |
| NOI Uplift | $[delta] ([X%] increase) |
| Going-in Cap Rate | [X.X%] |
| Stabilized Yield on Cost | [X.X%] |

---

### Exit Assumption
| | |
|---|---|
| Exit Cap Rate | [X.X%] |
| Exit Value | $[amount] |
| Hold Period | [X years] |

---

### Return Profile (Deal Level)
| | |
|---|---|
| Levered IRR | [X.X%] |
| Unlevered IRR | [X.X%] |
| Equity Multiple | [X.Xx] |
| Cash-on-Cash (Year 1) | [X.X%] |

---

### Risk / Mitigant
- **[Risk 1]:** [Mitigant — one sentence]
- **[Risk 2]:** [Mitigant — one sentence]

---
*Source: {source_file} | Ingested: {ingested_at}*
"""


def generate_deal_review() -> dict[str, Any]:
    """
    Read the underwriting layer from SSOT and produce a structured
    acquisition summary in the fixed template format.
    """
    s = ssot.load_ssot()
    underwriting = s["layers"].get("underwriting")

    if not underwriting:
        return {
            "error": (
                "No underwriting layer in SSOT. Upload an acquisition "
                "underwriting model first."
            )
        }

    if not llm_available():
        return {"error": "OPENAI_API_KEY is not set."}

    # Apply the Deal Review profile filter — only pass relevant metrics
    filtered = filter_layer_metrics(underwriting, "deal_review")

    # Format metrics for the prompt — flat dict of name → value for clarity
    metrics_flat = {
        name: {
            "value": data["value"],
            "sheet": data.get("sheet"),
            "cell": data.get("cell"),
        }
        for name, data in filtered["metrics"].items()
        if data.get("value") is not None
    }

    user_prompt = TEMPLATE.format(
        metrics_json=json.dumps(metrics_flat, indent=2, default=str),
        source_file=underwriting.get("source_file", "Unknown"),
        ingested_at=underwriting.get("ingested_at", "Unknown"),
    )

    narrative = complete(SYSTEM_PROMPT, user_prompt, temperature=0.1)

    # Save the acquisition summary back to SSOT as a permanent record
    _memorialize_acquisition(s, narrative, filtered, underwriting)

    return {
        "scenario": "deal_review",
        "narrative": narrative,
        "data_used": {
            "layers": ["underwriting"],
            "source_files": [underwriting["source_file"]],
            "metric_count_extracted": underwriting["metric_count"],
            "metric_count_used": filtered["metric_count"],
        },
    }


def _memorialize_acquisition(
    s: dict[str, Any],
    narrative: str,
    filtered: dict[str, Any],
    underwriting: dict[str, Any],
) -> None:
    """
    Save the acquisition summary as a permanent record in the SSOT.
    This is the 'founding event' — once written, it should not be overwritten
    by re-running deal review. The original thesis is immutable.
    """
    # Only memorialize once — don't overwrite if already exists
    if s["layers"].get("acquisition_summary"):
        return

    acquisition_record = {
        "source_file": underwriting.get("source_file"),
        "ingested_at": underwriting.get("ingested_at"),
        "metric_count": filtered["metric_count"],
        "metrics": filtered["metrics"],
        "narrative": narrative,
    }

    s["layers"]["acquisition_summary"] = acquisition_record
    ssot.save_ssot(s)
