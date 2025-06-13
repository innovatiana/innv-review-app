import streamlit as st
import pandas as pd
from modules.qa_checks import run_all_quality_checks

st.set_page_config(page_title="Dataset QA Review", layout="wide")

st.title("🧠 LLM-powered Dataset QA Review Tool")

# Upload CSV or JSONL
uploaded_file = st.file_uploader("Upload a CSV or JSONL file", type=["csv", "jsonl"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]

    try:
        if file_ext == "csv":
            dataset = pd.read_csv(uploaded_file)
        elif file_ext == "jsonl":
            dataset = pd.read_json(uploaded_file, lines=True)
        else:
            st.error("Unsupported file format")
            st.stop()

        st.success(f"✅ File loaded with {len(dataset)} rows and {len(dataset.columns)} columns.")
        st.dataframe(dataset.head(10))

        st.markdown("---")
        st.markdown("### ✅ Select Quality Checks to Run")

        # Dynamically list the LLM-powered checks
        available_checks = [
            "NER Span Conflicts (LLM)",
            "LLM Prompt/Response Validation (LLM)",
            "Timestamp Validation (LLM)",
            "Bounding Box Consistency (LLM)"
        ]

        selected_checks = st.multiselect("Choose QA checks to run", available_checks)

        if st.button("🚀 Run Selected Checks"):
            with st.spinner("Running quality checks..."):
                report = run_all_quality_checks(dataset, selected_checks)

            st.markdown("## 🧾 Summary Report")

            if not report:
                st.info("No checks selected.")
            else:
                for section, result in report.items():
                    with st.expander(f"🔍 {section}", expanded=True):
                        st.markdown(result)

    except Exception as e:
        st.error(f"❌ Error while reading file: {str(e)}")
