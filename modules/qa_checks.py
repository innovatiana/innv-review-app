import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client (replace with st.secrets if needed)
client = OpenAI(api_key="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")

def ask_openai(prompt, sample_rows):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a dataset quality control expert."},
                {"role": "user", "content": f"{prompt}\n\nSample rows:\n{sample_rows}"}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

def check_ner_conflicts_with_llm(df):
    sample_rows = df.head(5).to_json(orient="records", lines=True)
    prompt = (
        "Please review the following JSONL-formatted named entity recognition (NER) annotations. "
        "Identify if any spans are overlapping, conflicting, or incorrectly labeled. "
        "Return a list of potential issues per entry."
    )
    return ask_openai(prompt, sample_rows)

def check_prompt_response_with_llm(df):
    sample_rows = df.head(5).to_json(orient="records", lines=True)
    prompt = (
        "Each row contains a prompt and a response. Check for empty responses, overly generic answers, "
        "or lack of semantic connection between prompt and response. Return a quality note for each row."
    )
    return ask_openai(prompt, sample_rows)

def check_timestamps_with_llm(df):
    sample_rows = df.head(5).to_json(orient="records", lines=True)
    prompt = (
        "Each entry contains timestamps related to audio or video annotations. Check if formats are correct "
        "(e.g., ISO 8601 or float), start_time < end_time, and if there are overlaps or values beyond duration limits. "
        "Provide any detected issues."
    )
    return ask_openai(prompt, sample_rows)

def check_bboxes_with_llm(df):
    sample_rows = df.head(5).to_json(orient="records", lines=True)
    prompt = (
        "Review the bounding box annotations for each image. Validate if the boxes are within image dimensions, "
        "not overlapping where they shouldn't, and consistent with the label context. Return a summary of concerns."
    )
    return ask_openai(prompt, sample_rows)

def run_all_quality_checks(df, selected_checks):
    report = {}

    if "NER Span Conflicts (LLM)" in selected_checks:
        report["NER Span Conflicts (LLM)"] = check_ner_conflicts_with_llm(df)

    if "LLM Prompt/Response Validation (LLM)" in selected_checks:
        report["LLM Prompt/Response Validation (LLM)"] = check_prompt_response_with_llm(df)

    if "Timestamp Validation (LLM)" in selected_checks:
        report["Timestamp Validation (LLM)"] = check_timestamps_with_llm(df)

    if "Bounding Box Consistency (LLM)" in selected_checks:
        report["Bounding Box Consistency (LLM)"] = check_bboxes_with_llm(df)

    return report
