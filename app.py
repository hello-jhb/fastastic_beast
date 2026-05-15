import streamlit as st
from pathlib import Path

from document_extractor import extract_all_documents
from analysis_engine import generate_performance_analysis
from gpt_engine import ask_gpt, generate_asset_management_narrative
from file_classifier_gpt import classify_uploaded_files
from timeline_builder import build_investment_timeline


st.set_page_config(
    page_title="Real Estate AI Prototype",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Root font — sets the base for everything without touching spacing */
    html, body {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
    }

    /* App shell */
    .stApp, .main, section.main {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
    }

    /* Headers — no letter-spacing override to avoid word compression */
    h1, h2, h3, h4, h5, h6 {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-weight: 600;
        word-spacing: normal;
        letter-spacing: normal;
    }

    /* Markdown prose — explicit word/letter spacing reset to prevent compression */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-size: 14px;
        word-spacing: normal;
        letter-spacing: normal;
        line-height: 1.6;
    }

    /* Inline bold/italic inside markdown must not collapse surrounding spaces */
    .stMarkdown strong,
    .stMarkdown em,
    .stMarkdown b,
    .stMarkdown i {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        word-spacing: normal;
        letter-spacing: normal;
    }

    /* Captions and labels */
    .stCaption, label {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-size: 13px;
        word-spacing: normal;
        letter-spacing: normal;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] > div {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }

    /* Metric values */
    [data-testid="stMetricValue"] > div {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-size: 20px;
        font-weight: 600;
        word-spacing: normal;
        letter-spacing: normal;
    }

    /* Buttons */
    .stButton > button {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-weight: 500;
        letter-spacing: normal;
    }

    /* Dataframe */
    .stDataFrame {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-size: 13px;
    }

    /* JSON viewer — monospace only here */
    .stJson {
        font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
        font-size: 12px;
    }

    /* Expander header */
    [data-testid="stExpander"] summary {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        font-weight: 500;
        letter-spacing: normal;
    }

    /* Sidebar */
    [data-testid="stSidebar"] * {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)


def format_label(value):
    """Convert snake_case or underscore strings to Title Case for display."""
    if not value:
        return "Unknown"
    return str(value).replace("_", " ").title()


def safe_md(text):
    """
    Escape characters that Streamlit's markdown renderer would mis-interpret.
    Specifically, escape `$` so paired dollar signs in prose (e.g. '$25M ... $30M')
    don't get treated as LaTeX math delimiters and collapse the text between them.
    """
    if not text:
        return ""
    return str(text).replace("$", r"\$")


st.title("Real Estate AI Prototype")
st.caption(
    "Upload institutional real estate files → classify documents → reconstruct timeline → generate asset management intelligence"
)


# ---------------------------------------------------
# Session state
# ---------------------------------------------------

for key in [
    "classification_result",
    "timeline_result",
    "flexible_result",
    "analysis",
    "narrative"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------
# Directories
# ---------------------------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPOSITORY_DIR = Path("repository")
REPOSITORY_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# Upload files
# ---------------------------------------------------

st.header("1. Upload Investment Files")

uploaded_files = st.file_uploader(
    "Upload underwriting, business plans, financial statements, rent rolls, debt files, or reports",
    type=["xlsx", "xlsm", "csv", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIR / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"{len(uploaded_files)} file(s) uploaded.")


# ---------------------------------------------------
# Run analysis
# ---------------------------------------------------

st.header("2. Reconstruct Investment State")

if st.button("Run Analysis"):

    with st.spinner("Understanding uploaded files..."):
        classification_result = classify_uploaded_files("uploads")

    with st.spinner("Building investment timeline..."):
        timeline_result = build_investment_timeline(classification_result)

    with st.spinner("Extracting investment evidence..."):
        flexible_result = extract_all_documents("uploads", classification_result=classification_result)

    with st.spinner("Building investment context..."):
        analysis = generate_performance_analysis(flexible_result)

        # Add timeline/classification context to analysis package for GPT
        analysis["file_classification"] = classification_result
        analysis["investment_timeline"] = timeline_result

    with st.spinner("Generating asset management assessment..."):
        narrative = generate_asset_management_narrative(analysis)

    st.session_state.classification_result = classification_result
    st.session_state.timeline_result = timeline_result
    st.session_state.flexible_result = flexible_result
    st.session_state.analysis = analysis
    st.session_state.narrative = narrative

    st.success("Investment state reconstructed.")


# ---------------------------------------------------
# File understanding
# ---------------------------------------------------

if st.session_state.classification_result:

    st.header("3. File Understanding")

    classifications = st.session_state.classification_result.get(
        "classifications",
        []
    )

    for item in classifications:

        with st.container(border=True):

            st.markdown(
                f"### {item.get('file_name', 'Unknown file')}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Document Type",
                format_label(item.get("document_type"))
            )

            col2.metric(
                "Investment Role",
                format_label(item.get("investment_lifecycle_role"))
            )

            col3.metric(
                "Likely Period",
                str(item.get("likely_year", "Unknown"))
            )

            col4.metric(
                "Confidence",
                format_label(item.get("confidence"))
            )

            if item.get("relevant_tabs"):
                st.write(
                    "**Relevant Tabs:** "
                    + ", ".join(item.get("relevant_tabs", []))
                )

            if item.get("key_detected_sections"):
                st.write(
                    "**Detected Sections:** "
                    + ", ".join(item.get("key_detected_sections", []))
                )

            if item.get("reasoning"):
                st.caption(item.get("reasoning"))


# ---------------------------------------------------
# Investment timeline
# ---------------------------------------------------

if st.session_state.timeline_result:

    st.header("4. Investment Timeline")

    timeline = st.session_state.timeline_result
    events = timeline.get("timeline_events", [])

    if events:
        for event in events:
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 3])

                col1.metric(
                    "Period",
                    str(event.get("year", "Unknown"))
                )

                col2.write(
                    f"**{event.get('timeline_type', 'Unknown')}**"
                )

                col3.write(
                    event.get("description", "")
                )

                st.caption(
                    f"Source: {event.get('file_name', 'Unknown file')} | "
                    f"Confidence: {event.get('confidence', 'Unknown')}"
                )
    else:
        st.info("No timeline events were reconstructed.")


# ---------------------------------------------------
# Asset management assessment
# ---------------------------------------------------

if st.session_state.narrative:

    st.header("5. Asset Management Assessment")

    # Compact strip showing which files the assessment is grounded in
    if st.session_state.classification_result:
        classifications = st.session_state.classification_result.get("classifications", [])
        if classifications:
            file_chips = " · ".join(
                f"**{item.get('file_name', '?')}** ({format_label(item.get('document_type'))})"
                for item in classifications
            )
            st.caption("Sources reviewed: " + file_chips)

    with st.container(border=True):
        st.markdown(safe_md(st.session_state.narrative))


# ---------------------------------------------------
# Ask questions
# ---------------------------------------------------

if st.session_state.analysis and st.session_state.flexible_result:

    st.header("6. Ask the Asset")

    question = st.chat_input(
        "Ask a question about the uploaded investment files..."
    )

    if question:

        st.markdown(f"**Question:** {question}")

        with st.spinner("Analyzing investment context..."):

            answer = ask_gpt(
                question,
                st.session_state.flexible_result,
                st.session_state.analysis
            )

        st.markdown("### Answer")
        with st.container(border=True):
            st.markdown(safe_md(answer))


# ---------------------------------------------------
# Debug / advanced diagnostics
# ---------------------------------------------------

if (
    st.session_state.flexible_result
    or st.session_state.analysis
    or st.session_state.classification_result
    or st.session_state.timeline_result
):

    with st.expander("Debug / Advanced Diagnostics"):

        if st.session_state.flexible_result:

            st.subheader("Metric Extraction Coverage")

            flexible_result = st.session_state.flexible_result

            total_metrics = flexible_result.get("total_metrics", 0)
            extracted_metrics = flexible_result.get("extracted_count", 0)
            missing_metrics = flexible_result.get("missing_count", 0)

            col1, col2, col3 = st.columns(3)

            col1.metric("Catalog Metrics", total_metrics)
            col2.metric("Metrics Found", extracted_metrics)
            col3.metric("Metrics Missing", missing_metrics)

            progress_value = (
                extracted_metrics / total_metrics
                if total_metrics > 0
                else 0
            )

            st.progress(progress_value)

            with st.expander("View Extracted Metrics"):
                st.dataframe(
                    flexible_result.get("extracted_metrics", [])
                )

            with st.expander("View Missing Metrics"):
                st.dataframe(
                    flexible_result.get("missing_metrics", [])
                )

        if st.session_state.classification_result:
            st.subheader("File Classification JSON")
            st.json(st.session_state.classification_result)

        if st.session_state.timeline_result:
            st.subheader("Investment Timeline JSON")
            st.json(st.session_state.timeline_result)

        if st.session_state.analysis:
            st.subheader("Structured Analysis Context")
            st.json(st.session_state.analysis)

        if st.session_state.flexible_result:
            st.subheader("Flexible Extraction Output")
            st.json(st.session_state.flexible_result)