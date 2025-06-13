import streamlit as st
import pandas as pd
from modules.qa_checks import run_all_quality_checks

st.set_page_config(page_title="Dataset QA Review", layout="wide")

st.title("📊 Dataset Quality Review Tool")

uploaded_file = st.file_uploader("Upload a dataset file (CSV, JSON, JSONL)", type=["csv", "json", "jsonl"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            dataset = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".jsonl"):
            dataset = pd.read_json(uploaded_file, lines=True)
        elif uploaded_file.name.endswith(".json"):
            dataset = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success("✅ Dataset loaded successfully!")
        st.write("### Preview", dataset.head())

        if st.button("Run Quality Checks"):
            with st.spinner("Running quality checks..."):
                report = run_all_quality_checks(dataset)

            st.markdown("## ✅ Summary Report")
            for section, result in report.items():
                st.subheader(f"🔍 {section.replace('_', ' ').title()}")
                st.code(result if isinstance(result, str) else str(result))

    except Exception as e:
        st.error(f"❌ Error while reading file: {str(e)}")
