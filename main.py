import streamlit as st
import pandas as pd
import json
from modules.qa_checks import run_all_quality_checks

st.set_page_config(page_title="📊 AI Dataset QA Review", layout="wide")
st.title("🧪 Dataset Quality Review Tool")

st.markdown("""
Upload an annotated dataset in **CSV, JSON, JSONL or XML** format. This tool will analyze the file and perform quality checks automatically, based on the structure of your data.
""")

uploaded_file = st.file_uploader("📁 Upload your dataset file", type=["csv", "json", "jsonl", "xml"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "csv":
            dataset = pd.read_csv(uploaded_file)
        elif file_type == "jsonl":
            dataset = pd.read_json(uploaded_file, lines=True)
        elif file_type == "json":
            try:
                raw = json.load(uploaded_file)
                if isinstance(raw, list):
                    dataset = pd.DataFrame(raw)
                elif isinstance(raw, dict):
                    dataset = pd.json_normalize(raw)
                else:
                    st.error("❌ Unsupported JSON structure.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Failed to parse JSON: {e}")
                st.stop()
        elif file_type == "xml":
            dataset = pd.read_xml(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error while reading file: {e}")
        st.stop()

    st.success("✅ Dataset loaded successfully!")

    st.subheader("📋 Dataset Columns")
    st.write(list(dataset.columns))

    st.markdown("---")
    st.subheader("🧪 Available Quality Checks")

    checks_available = []

    if {"text", "label"}.issubset(dataset.columns):
        checks_available += [
            ("🧠 Cleanlab anomaly detection (classification)", "Detects potential mislabels using statistical learning."),
            ("📊 Class imbalance / label distribution", "Shows how labels are distributed, and highlights imbalances."),
            ("🗣️ Token frequency & text length", "Identifies very short/long texts and overrepresented tokens."),
            ("🌍 Language detection", "Checks whether the detected language matches the expected one."),
        ]
    if {"start", "end"}.issubset(dataset.columns):
        checks_available += [("🧵 NER span overlap / conflict", "Detects overlapping entity spans and conflicting labels.")]
    if {"prompt", "response"}.issubset(dataset.columns):
        checks_available += [("🤖 Prompt/Response validation (LLM)", "Verifies that both fields are filled, with valid lengths and similarity.")]
    if {"bbox_x", "bbox_y", "bbox_width", "bbox_height"}.issubset(dataset.columns):
        checks_available += [("🖼️ Bounding box consistency", "Checks if bounding boxes are valid and within image dimensions.")]
    if {"start_time", "end_time"}.issubset(dataset.columns):
        checks_available += [("🕒 Timestamp validation", "Verifies that timestamps are correctly ordered and formatted.")]
    if len(dataset.columns) > 0:
        checks_available += [
            ("🧼 Missing values", "Identifies missing or null values in key columns."),
            ("🧾 Duplicates", "Detects duplicate or near-duplicate rows.")
        ]

    for check, explanation in checks_available:
        st.markdown(f"- {check}\n    > 📌 {explanation}")

    st.markdown("---")
    st.subheader("📈 Run Automated Analysis")

    if st.button("▶️ Run Quality Checks"):
        with st.spinner("Analyzing dataset..."):
            metadata = {}  # Reserved for future use (e.g., media files or label schema)
            report = run_all_quality_checks(dataset, metadata)

        st.success("✅ Analysis complete!")

        st.subheader("📌 Summary Report")
        for key, value in report.items():
            st.markdown(f"**{key}**: {value}")
else:
    st.info("⬆️ Please upload a dataset file to begin.")
