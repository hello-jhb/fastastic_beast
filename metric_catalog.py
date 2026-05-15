from pathlib import Path
import pandas as pd
import re


CATALOG_PATH = Path("Snapshot Metric.xlsx")
REPOSITORY_DIR = Path("repository")


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("_", " ")
    )


def make_metric_id(metric_name):
    text = metric_name.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def split_list(value):
    text = clean_text(value)
    if not text:
        return []

    parts = re.split(r";|,|\n", text)
    return [p.strip() for p in parts if p.strip()]


def get_col(row, col_map, possible_names):
    for name in possible_names:
        key = normalize_column_name(name)
        if key in col_map:
            return clean_text(row[col_map[key]])
    return ""


# -----------------------------
# Inference logic
# -----------------------------
def infer_priority(metric_name, category):
    text = f"{metric_name} {category}".lower()

    high_keywords = [
        "noi", "revenue", "expense", "opex", "dscr", "debt",
        "irr", "equity multiple", "basis", "value", "valuation",
        "occupancy", "cap rate", "capex", "cash flow"
    ]

    for keyword in high_keywords:
        if keyword in text:
            return "High"

    return "Medium"


def infer_layers(source_type, metric_name):
    text = f"{source_type} {metric_name}".lower()
    layers = []

    if any(x in text for x in ["acquisition", "underwriting"]):
        layers.append("Acquisition Underwriting")
    if any(x in text for x in ["business plan", "budget", "bp", "forecast"]):
        layers.append("Business Plan")
    if any(x in text for x in ["actual", "financial statement", "ledger", "gl"]):
        layers.append("Actuals")
    if any(x in text for x in ["rent roll", "lease", "tenant", "occupancy", "walt"]):
        layers.append("Leasing")
    if any(x in text for x in ["debt", "loan", "dscr", "ltv"]):
        layers.append("Debt")
    if any(x in text for x in ["capex", "capital"]):
        layers.append("CapEx")

    return layers or ["General"]


def build_aliases(metric_name, aliases_text):
    aliases = []

    if metric_name:
        aliases.append(metric_name)

    aliases += split_list(aliases_text)

    lower = metric_name.lower()

    if "noi" in lower or "net operating income" in lower:
        aliases += ["NOI", "Net Operating Income"]

    if "revenue" in lower or "income" in lower:
        aliases += [
            "Revenue",
            "Total Revenue",
            "Total Operating Revenue",
            "Effective Gross Revenue",
            "EGI"
        ]

    if "expense" in lower or "opex" in lower:
        aliases += [
            "OpEx",
            "Operating Expenses",
            "Total Operating Expenses",
            "Expenses"
        ]

    if "dscr" in lower:
        aliases += ["DSCR", "Debt Service Coverage Ratio"]

    if "irr" in lower:
        aliases += ["IRR", "Internal Rate of Return"]

    if "levered irr" in lower:
        aliases += ["Levered IRR", "Equity IRR"]

    if "unlevered irr" in lower:
        aliases += ["Unlevered IRR", "Property IRR"]

    if "cap rate" in lower:
        aliases += ["Cap Rate", "Capitalization Rate", "Going-in Cap Rate", "Exit Cap Rate"]

    if "purchase price" in lower:
        aliases += ["Purchase Price", "Acquisition Price"]

    if "basis" in lower:
        aliases += ["Basis", "Total Basis", "Cost Basis"]

    if "occupancy" in lower:
        aliases += ["Occupancy", "Occupied", "Vacancy"]

    if "walt" in lower or "wale" in lower:
        aliases += ["WALT", "WALE", "Weighted Average Lease Term"]

    if "ltv" in lower:
        aliases += ["LTV", "Loan to Value"]

    if "debt" in lower or "loan" in lower:
        aliases += ["Debt", "Loan", "Loan Amount", "Debt Balance"]

    if "capex" in lower:
        aliases += ["CapEx", "Capital Expenditure", "Capital Costs"]

    # Deduplicate while preserving order
    cleaned = []
    seen = set()

    for alias in aliases:
        alias = clean_text(alias)
        if alias and alias.lower() not in seen:
            cleaned.append(alias)
            seen.add(alias.lower())

    return cleaned


# -----------------------------
# Main catalog loader
# -----------------------------
def load_metric_catalog(path=CATALOG_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metric catalog not found: {path}. "
            "Save 'Snapshot Metric.xlsx' inside your real_estate_ai folder."
        )

    df = pd.read_excel(path)

    # Normalize column mapping
    col_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    catalog = []

    for _, row in df.iterrows():
        metric_name = get_col(row, col_map, ["Metric Name", "Metric", "Name"])

        if not metric_name:
            continue

        category = get_col(row, col_map, ["Category"])
        definition = get_col(row, col_map, ["Definition"])
        formula = get_col(row, col_map, ["Formula", "Calculation Method"])
        data_format = get_col(row, col_map, ["Data Format", "Format"])
        source_type = get_col(row, col_map, ["Source Type", "Source", "Required Source Type"])
        system = get_col(row, col_map, ["System"])
        stakeholder = get_col(row, col_map, ["Who Cares", "Stakeholder", "User"])
        aliases_text = get_col(row, col_map, ["Aliases", "Search Terms"])
        layer_text = get_col(row, col_map, ["Layer", "Layers"])
        core_question = get_col(row, col_map, ["Core Question", "Used For Core Question"])
        priority = get_col(row, col_map, ["Priority"])
        dashboard_flag = get_col(row, col_map, ["Dashboard Flag", "Dashboard", "Show on Dashboard"])
        extraction_notes = get_col(row, col_map, ["Extraction Notes", "Notes"])

        aliases = build_aliases(metric_name, aliases_text)

        layers = split_list(layer_text)
        if not layers:
            layers = infer_layers(source_type, metric_name)

        if not priority:
            priority = infer_priority(metric_name, category)

        metric = {
            "metric_id": make_metric_id(metric_name),
            "metric_name": metric_name,
            "category": category,
            "definition": definition,
            "formula": formula,
            "data_format": data_format,
            "source": source_type,
            "system": system,
            "stakeholder": stakeholder,
            "layers": layers,
            "aliases": aliases,
            "core_question": core_question,
            "priority": priority,
            "dashboard_flag": dashboard_flag,
            "extraction_notes": extraction_notes,
        }

        catalog.append(metric)

    return catalog


def catalog_to_dataframe(catalog):
    rows = []

    for item in catalog:
        rows.append({
            "metric_id": item["metric_id"],
            "metric_name": item["metric_name"],
            "category": item["category"],
            "definition": item["definition"],
            "formula": item["formula"],
            "data_format": item["data_format"],
            "source": item["source"],
            "system": item["system"],
            "stakeholder": item["stakeholder"],
            "layers": "; ".join(item["layers"]),
            "aliases": "; ".join(item["aliases"]),
            "core_question": item["core_question"],
            "priority": item["priority"],
            "dashboard_flag": item["dashboard_flag"],
            "extraction_notes": item["extraction_notes"],
        })

    return pd.DataFrame(rows)


def save_catalog_preview(output_path="repository/metric_catalog_preview.csv"):
    catalog = load_metric_catalog()
    df = catalog_to_dataframe(catalog)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    df.to_csv(output_path, index=False)

    return output_path


# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":
    catalog = load_metric_catalog()
    print(f"Loaded {len(catalog)} metrics.")

    preview_path = save_catalog_preview()
    print(f"Saved catalog preview to: {preview_path}")