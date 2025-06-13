import streamlit as st
from modules.parser import load_dataset
from modules.qa_checks import run_all_quality_checks
from modules.visualization import display_summary, display_detailed_issues

st.set_page_config(page_title="AI Dataset QA Review Tool", layout="wide")
st.title("📊 AI Dataset Quality Review")

uploaded_file = st.file_uploader("Upload a dataset file (.csv, .json, .jsonl, .xml, .zip)", type=["csv", "json", "jsonl", "xml", "zip"])

if uploaded_file:
    with st.spinner("Loading and parsing the dataset..."):
        dataset, metadata = load_dataset(uploaded_file)

    if dataset is not None:
        st.success("✅ Dataset loaded successfully!")

        with st.expander("View Raw Data"):
            st.dataframe(dataset.head(100))

        with st.spinner("Running quality checks..."):
            report = run_all_quality_checks(dataset, metadata)

        st.success("🔍 Quality checks completed.")

        display_summary(report)
        display_detailed_issues(report)
    else:
        st.error("❌ Could not parse the dataset. Check the format or content.")
