import streamlit as st
from pathlib import Path

from flexible_extractor import scan_uploaded_files
from analysis_engine import generate_performance_analysis
from gpt_engine import ask_gpt, generate_asset_management_narrative


st.set_page_config(
    page_title="Real Estate AI Prototype",
    layout="wide"
)

st.title("Real Estate AI Prototype")
st.caption("Upload files → extract asset-state metrics → assess core questions → generate AM intelligence")


for key in ["flexible_result", "analysis", "narrative"]:
    if key not in st.session_state:
        st.session_state[key] = None


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPOSITORY_DIR = Path("repository")
REPOSITORY_DIR.mkdir(exist_ok=True)


st.header("1. Upload Source Files")

uploaded_files = st.file_uploader(
    "Upload underwriting, business plan, financial statements, rent roll, debt files, or reports",
    type=["xlsx", "xlsm", "csv", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"{len(uploaded_files)} file(s) uploaded.")


st.header("2. Run Analysis")

if st.button("Run Extraction + Analysis"):

    with st.spinner("Scanning uploaded files against metric catalog..."):
        flexible_result = scan_uploaded_files("uploads")

    with st.spinner("Building analysis context..."):
        analysis = generate_performance_analysis(flexible_result)

    with st.spinner("Generating asset management narrative..."):
        narrative = generate_asset_management_narrative(analysis)

    st.session_state.flexible_result = flexible_result
    st.session_state.analysis = analysis
    st.session_state.narrative = narrative

    st.success("Analysis complete.")


if st.session_state.flexible_result:

    flexible_result = st.session_state.flexible_result

    st.header("3. Metric Extraction Coverage")

    col1, col2, col3 = st.columns(3)

    total_metrics = flexible_result.get("total_metrics", 0)
    extracted_metrics = flexible_result.get("extracted_count", 0)
    missing_metrics = flexible_result.get("missing_count", 0)

    col1.metric("Catalog Metrics", total_metrics)
    col2.metric("Metrics Found", extracted_metrics)
    col3.metric("Metrics Missing", missing_metrics)

    progress_value = extracted_metrics / total_metrics if total_metrics > 0 else 0
    st.progress(progress_value)

    if total_metrics == 0:
        st.warning("Metric catalog did not load correctly.")

    if extracted_metrics == 0:
        st.warning(
            "No recognizable snapshot metrics were extracted. "
            "The uploaded file may be blank, unsupported, or missing relevant real estate data."
        )

    with st.expander("View Extracted Metrics"):
        st.dataframe(flexible_result.get("extracted_metrics", []))

    with st.expander("View Missing Metrics"):
        st.dataframe(flexible_result.get("missing_metrics", []))


if st.session_state.narrative:

    st.header("4. Asset Management Assessment")
    st.markdown(st.session_state.narrative)


if st.session_state.analysis:

    with st.expander("View Structured Analysis Context"):
        st.json(st.session_state.analysis)

    with st.expander("View Flexible Metric Scan Output"):
        st.json(st.session_state.flexible_result)


if st.session_state.analysis and st.session_state.flexible_result:

    st.header("5. Ask the Asset")

    question = st.chat_input("Ask a question about the uploaded files...")

    if question:
        st.markdown(f"**Question:** {question}")

        with st.spinner("Thinking..."):
            answer = ask_gpt(
                question,
                st.session_state.flexible_result,
                st.session_state.analysis
            )

        st.markdown("### Answer")
        st.write(answer)
