# analysis_engine.py

def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def get_extracted_metrics(flexible_result):
    if not flexible_result:
        return []
    return flexible_result.get("extracted_metrics", []) or []


def safe_bool(value):
    return bool(value)


def metric_matches(item, keywords):
    """
    Checks whether a metric item appears to match any keyword.
    Searches metric name, category, source file, sheet, matched alias, and definition.
    """
    text = " ".join([
        normalize_text(item.get("metric_name")),
        normalize_text(item.get("category")),
        normalize_text(item.get("source_file")),
        normalize_text(item.get("sheet")),
        normalize_text(item.get("matched_alias")),
        normalize_text(item.get("definition")),
    ])

    return any(normalize_text(keyword) in text for keyword in keywords)


def has_metric(metrics, keywords):
    return any(metric_matches(item, keywords) for item in metrics)


def has_layer_signal(metrics, layer_keywords):
    """
    Detects source/layer signal based on source file, sheet, metric name, or category.
    This is imperfect, but useful for prototype-level relationship awareness.
    """
    for item in metrics:
        text = " ".join([
            normalize_text(item.get("source_file")),
            normalize_text(item.get("sheet")),
            normalize_text(item.get("metric_name")),
            normalize_text(item.get("category")),
        ])

        for keyword in layer_keywords:
            if normalize_text(keyword) in text:
                return True

    return False


def relationship_check(metrics):
    """
    Relationship-aware evidence checks.
    This is stronger than simple metric presence because it checks whether
    source layers appear to exist alongside relevant metric types.
    """

    # -----------------------------
    # Layer signals
    # -----------------------------
    layer_signals = {
        "actuals_present": has_layer_signal(
            metrics,
            ["actual", "financial statement", "fs", "t12", "monthly", "ledger", "2021", "2022", "2023"]
        ),
        "business_plan_or_forecast_present": has_layer_signal(
            metrics,
            ["business plan", "budget", "bp", "forecast", "proforma", "projection"]
        ),
        "acquisition_underwriting_present": has_layer_signal(
            metrics,
            ["acquisition", "underwriting", "uw", "purchase"]
        ),
        "debt_source_present": has_layer_signal(
            metrics,
            ["debt", "loan", "dscr", "lender", "financing"]
        ),
        "leasing_source_present": has_layer_signal(
            metrics,
            ["rent roll", "lease", "tenant", "rollover", "expiration", "walt", "wale"]
        ),
        "capex_source_present": has_layer_signal(
            metrics,
            ["capex", "capital", "cost to complete", "renovation"]
        ),
    }

    # -----------------------------
    # Metric signals
    # -----------------------------
    metric_signals = {
        "noi": has_metric(metrics, ["noi", "net operating income"]),
        "revenue": has_metric(metrics, ["revenue", "income", "effective gross revenue", "egi"]),
        "expense": has_metric(metrics, ["expense", "opex", "operating expense"]),
        "occupancy": has_metric(metrics, ["occupancy", "occupied", "vacancy"]),
        "variance": has_metric(metrics, ["variance", "budget vs actual", "actual vs budget"]),

        "dscr": has_metric(metrics, ["dscr", "debt service coverage"]),
        "debt_yield": has_metric(metrics, ["debt yield"]),
        "ltv": has_metric(metrics, ["ltv", "loan to value"]),
        "debt_service": has_metric(metrics, ["debt service", "loan payment", "principal and interest"]),
        "debt_balance": has_metric(metrics, ["debt balance", "loan balance", "outstanding debt"]),

        "capex": has_metric(metrics, ["capex", "capital expenditure", "capital cost", "capital costs"]),
        "capex_roi": has_metric(metrics, ["capex roi", "return on capex"]),
        "yield_on_cost": has_metric(metrics, ["yield on cost"]),
        "incremental_noi": has_metric(metrics, ["incremental noi"]),
        "cost_to_complete": has_metric(metrics, ["cost to complete", "remaining cost"]),

        "basis": has_metric(metrics, ["basis", "cost basis", "total basis", "purchase price", "acquisition price"]),
        "value": has_metric(metrics, ["value", "valuation", "market value", "implied value"]),
        "cap_rate": has_metric(metrics, ["cap rate", "capitalization rate"]),
        "irr": has_metric(metrics, ["irr", "internal rate"]),
        "equity_multiple": has_metric(metrics, ["equity multiple", "multiple"]),

        "walt": has_metric(metrics, ["walt", "wale", "weighted average lease"]),
        "tenant_concentration": has_metric(metrics, ["tenant concentration", "top tenant", "largest tenant"]),
        "delinquency": has_metric(metrics, ["delinquency", "bad debt", "credit loss", "collection"]),
        "rollover": has_metric(metrics, ["rollover", "expiration", "lease expiry", "lease expiration"]),
    }

    return {
        "layer_signals": layer_signals,
        "metric_signals": metric_signals,
    }


def assess_question(question, required_metrics, available_metrics, relationship_required, relationship_met, limitations):
    available = [x for x in available_metrics if x]
    missing = [x for x in required_metrics if x not in available]

    denominator = len(required_metrics) if required_metrics else 1
    coverage_pct = len(available) / denominator

    if relationship_required and not relationship_met:
        if coverage_pct >= 0.6:
            coverage = "partial"
        else:
            coverage = "low"
    else:
        if coverage_pct >= 0.75:
            coverage = "high"
        elif coverage_pct >= 0.4:
            coverage = "partial"
        else:
            coverage = "low"

    return {
        "question": question,
        "coverage": coverage,
        "available_metrics": available,
        "missing_metrics": missing,
        "coverage_pct": round(coverage_pct, 2),
        "relationship_required": relationship_required,
        "relationship_met": bool(relationship_met),
        "limitations": limitations or [],
    }


def core_question_coverage(flexible_result):
    metrics = get_extracted_metrics(flexible_result)
    signals = relationship_check(metrics)

    layers = signals.get("layer_signals", {})
    m = signals.get("metric_signals", {})

    results = []

    # ---------------------------------------------------
    # 1. Are we performing vs plan?
    # ---------------------------------------------------
    required = [
        "NOI",
        "Revenue",
        "Expense",
        "Occupancy",
        "Variance / Budget vs Actual",
    ]

    available = [
        "NOI" if m.get("noi", False) else None,
        "Revenue" if m.get("revenue", False) else None,
        "Expense" if m.get("expense", False) else None,
        "Occupancy" if m.get("occupancy", False) else None,
        "Variance / Budget vs Actual" if m.get("variance", False) else None,
    ]

    relationship_met = (
        layers.get("actuals_present", False)
        and layers.get("business_plan_or_forecast_present", False)
        and (
            m.get("noi", False)
            or m.get("revenue", False)
            or m.get("expense", False)
        )
    )

    limitations = []
    if not layers.get("actuals_present", False):
        limitations.append("Actual operating data was not clearly detected.")
    if not layers.get("business_plan_or_forecast_present", False):
        limitations.append("Business plan, budget, or forecast data was not clearly detected.")
    if not relationship_met:
        limitations.append(
            "Performance vs plan requires actuals and plan/budget data for comparable periods."
        )

    results.append(assess_question(
        "Are we performing vs plan?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    # ---------------------------------------------------
    # 2. Is the income durable?
    # ---------------------------------------------------
    required = [
        "WALT / WALE",
        "Occupancy",
        "Tenant Concentration",
        "Delinquency / Bad Debt",
        "Rollover / Expiration",
    ]

    available = [
        "WALT / WALE" if m.get("walt", False) else None,
        "Occupancy" if m.get("occupancy", False) else None,
        "Tenant Concentration" if m.get("tenant_concentration", False) else None,
        "Delinquency / Bad Debt" if m.get("delinquency", False) else None,
        "Rollover / Expiration" if m.get("rollover", False) else None,
    ]

    relationship_met = (
        layers.get("leasing_source_present", False)
        and (
            m.get("walt", False)
            or m.get("rollover", False)
            or m.get("tenant_concentration", False)
            or m.get("occupancy", False)
        )
    )

    limitations = []
    if not layers.get("leasing_source_present", False):
        limitations.append("Rent roll, lease, or tenant-level source was not clearly detected.")
    if not relationship_met:
        limitations.append(
            "Income durability requires lease structure, rollover, tenant concentration, or tenant health data."
        )

    results.append(assess_question(
        "Is the income durable?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    # ---------------------------------------------------
    # 3. Is the leverage healthy?
    # ---------------------------------------------------
    required = [
        "DSCR",
        "Debt Yield",
        "LTV",
        "Debt Balance",
        "Debt Service",
    ]

    available = [
        "DSCR" if m.get("dscr", False) else None,
        "Debt Yield" if m.get("debt_yield", False) else None,
        "LTV" if m.get("ltv", False) else None,
        "Debt Balance" if m.get("debt_balance", False) else None,
        "Debt Service" if m.get("debt_service", False) else None,
    ]

    relationship_met = (
        layers.get("debt_source_present", False)
        and (
            m.get("dscr", False)
            or m.get("debt_yield", False)
            or m.get("ltv", False)
            or m.get("debt_service", False)
            or m.get("debt_balance", False)
        )
    )

    limitations = []
    if not layers.get("debt_source_present", False):
        limitations.append("Debt model, loan statement, or debt schedule was not clearly detected.")
    if not relationship_met:
        limitations.append(
            "Leverage health requires debt terms and operating income / coverage metrics."
        )

    results.append(assess_question(
        "Is the leverage healthy?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    # ---------------------------------------------------
    # 4. Is further capital justified?
    # ---------------------------------------------------
    required = [
        "CapEx",
        "CapEx ROI",
        "Yield on Cost",
        "Incremental NOI",
        "Cost to Complete",
    ]

    available = [
        "CapEx" if m.get("capex", False) else None,
        "CapEx ROI" if m.get("capex_roi", False) else None,
        "Yield on Cost" if m.get("yield_on_cost", False) else None,
        "Incremental NOI" if m.get("incremental_noi", False) else None,
        "Cost to Complete" if m.get("cost_to_complete", False) else None,
    ]

    relationship_met = (
        m.get("capex", False)
        and (
            m.get("incremental_noi", False)
            or m.get("capex_roi", False)
            or m.get("yield_on_cost", False)
            or m.get("cost_to_complete", False)
        )
    )

    limitations = []
    if not m.get("capex", False):
        limitations.append("CapEx spend or budget was not detected.")
    if not relationship_met:
        limitations.append(
            "Capital justification requires linking capital spend to incremental NOI, yield on cost, or value creation."
        )

    results.append(assess_question(
        "Is further capital justified?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    # ---------------------------------------------------
    # 5. Is the asset worth its basis?
    # ---------------------------------------------------
    required = [
        "Basis",
        "Value",
        "Cap Rate",
        "IRR",
        "Equity Multiple",
    ]

    available = [
        "Basis" if m.get("basis", False) else None,
        "Value" if m.get("value", False) else None,
        "Cap Rate" if m.get("cap_rate", False) else None,
        "IRR" if m.get("irr", False) else None,
        "Equity Multiple" if m.get("equity_multiple", False) else None,
    ]

    relationship_met = (
        (
            layers.get("acquisition_underwriting_present", False)
            or m.get("basis", False)
        )
        and (
            m.get("value", False)
            or m.get("cap_rate", False)
            or m.get("irr", False)
            or m.get("equity_multiple", False)
        )
    )

    limitations = []
    if not layers.get("acquisition_underwriting_present", False) and not m.get("basis", False):
        limitations.append("Acquisition basis or cost basis was not clearly detected.")
    if not relationship_met:
        limitations.append(
            "Worth-basis analysis requires basis plus valuation or return metrics."
        )

    results.append(assess_question(
        "Is the asset worth its basis?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    # ---------------------------------------------------
    # 6. Is risk increasing or decreasing over time?
    # ---------------------------------------------------
    required = [
        "NOI / NOI Trend",
        "Revenue / Revenue Trend",
        "Expense / Expense Trend",
        "DSCR / DSCR Trend",
        "Occupancy / Occupancy Trend",
    ]

    available = [
        "NOI / NOI Trend" if m.get("noi", False) else None,
        "Revenue / Revenue Trend" if m.get("revenue", False) else None,
        "Expense / Expense Trend" if m.get("expense", False) else None,
        "DSCR / DSCR Trend" if m.get("dscr", False) else None,
        "Occupancy / Occupancy Trend" if m.get("occupancy", False) else None,
    ]

    relationship_met = (
        layers.get("actuals_present", False)
        and (
            m.get("noi", False)
            or m.get("revenue", False)
            or m.get("expense", False)
            or m.get("dscr", False)
            or m.get("occupancy", False)
        )
    )

    limitations = []
    if not layers.get("actuals_present", False):
        limitations.append("Actual time-series data was not clearly detected.")
    if not relationship_met:
        limitations.append(
            "Risk trend analysis requires metrics across time, not just a single static data point."
        )

    results.append(assess_question(
        "Is risk increasing or decreasing over time?",
        required,
        available,
        relationship_required=True,
        relationship_met=relationship_met,
        limitations=limitations,
    ))

    return results


def summarize_extracted_metrics(flexible_result, limit=80):
    extracted = get_extracted_metrics(flexible_result)
    simplified = []

    for item in extracted[:limit]:
        simplified.append({
            "metric_name": item.get("metric_name"),
            "category": item.get("category"),
            "value": item.get("value"),
            "source_file": item.get("source_file"),
            "sheet": item.get("sheet"),
            "value_cell": item.get("value_cell"),
            "confidence": item.get("confidence"),
        })

    return simplified


def summarize_missing_metrics(flexible_result, limit=80):
    if not flexible_result:
        return []

    missing = flexible_result.get("missing_metrics", []) or []

    high_priority = [
        item for item in missing
        if normalize_text(item.get("priority")) == "high"
    ]

    selected = high_priority[:limit] if high_priority else missing[:limit]
    simplified = []

    for item in selected:
        simplified.append({
            "metric_name": item.get("metric_name"),
            "category": item.get("category"),
            "definition": item.get("definition"),
            "priority": item.get("priority"),
            "source": item.get("source"),
        })

    return simplified


def assess_file_signal(flexible_result):
    if not flexible_result:
        return {
            "status": "no_result",
            "message": "No extraction result was provided."
        }

    total_metrics = flexible_result.get("total_metrics", 0) or 0
    extracted_count = flexible_result.get("extracted_count", 0) or 0

    if total_metrics == 0:
        return {
            "status": "catalog_error",
            "message": "Metric catalog did not load correctly."
        }

    if extracted_count == 0:
        return {
            "status": "no_metrics_found",
            "message": (
                "No recognizable real estate metrics were extracted from the uploaded files. "
                "The files may be blank, unsupported, or not relevant to the current metric catalog."
            )
        }

    if extracted_count < 5:
        return {
            "status": "very_limited_data",
            "message": (
                "Only a small number of metrics were extracted. "
                "Analysis should be treated as highly preliminary."
            )
        }

    return {
        "status": "metrics_found",
        "message": "Metric extraction produced enough signal for preliminary analysis."
    }


def generate_performance_analysis(flexible_result):
    metrics = get_extracted_metrics(flexible_result)

    coverage = core_question_coverage(flexible_result)
    file_signal = assess_file_signal(flexible_result)
    relationships = relationship_check(metrics)

    analysis_context = {
        "analysis_mode": "generalized_relationship_aware_extraction",
        "file_signal": file_signal,
        "metric_catalog_coverage": {
            "total_metrics": flexible_result.get("total_metrics", 0) if flexible_result else 0,
            "extracted_count": flexible_result.get("extracted_count", 0) if flexible_result else 0,
            "missing_count": flexible_result.get("missing_count", 0) if flexible_result else 0,
        },
        "relationship_signals": relationships,
        "core_question_coverage": coverage,
        "extracted_metrics_sample": summarize_extracted_metrics(flexible_result),
        "missing_metrics_sample": summarize_missing_metrics(flexible_result),
        "instruction_to_gpt": (
            "Generate preliminary asset management analysis only from available extracted metrics. "
            "If data is insufficient, say so clearly. Explain what can and cannot be assessed. "
            "Do not invent financial values or assume missing underwriting, business plan, actual, debt, or leasing data. "
            "Pay attention to whether relationships exist, not just whether individual metrics are present."
        ),
    }

    return analysis_context
