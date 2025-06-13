import openai
import pandas as pd
import json
import streamlit as st

openai.api_key = "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"

def ask_openai(prompt, sample_rows):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a dataset quality control expert."},
                {"role": "user", "content": f"{prompt}\n\nSample rows:\n{sample_rows}"}
            ],
            temperature=0,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

# 1. NER Conflict Check
def check_ner_conflicts_with_llm(df, sample_size=10):
    prompt = """
You are reviewing a dataset annotated for Named Entity Recognition (NER).
Please identify:
- Any overlapping or conflicting spans (same start/end, different labels)
- Duplicate entities
- Any invalid or suspicious annotation patterns
Return your findings in Markdown bullet points.
"""
    sample = df.sample(min(sample_size, len(df))).to_dict(orient="records")
    return ask_openai(prompt, sample)

# 2. LLM Prompt/Response
def check_prompt_response_with_llm(df, sample_size=10):
    prompt = """
This dataset contains prompt/response pairs used to train or evaluate a language model.
Please detect:
- Empty or overly short/long responses
- Prompts that match responses (copy-paste)
- Generic, low-effort responses
Return a list of issues and recommendations in Markdown.
"""
    sample = df.sample(min(sample_size, len(df))).to_dict(orient="records")
    return ask_openai(prompt, sample)

# 3. Timestamp Validation (audio/video)
def check_timestamps_with_llm(df, sample_size=10):
    prompt = """
You are checking a dataset containing timestamped segments for audio or video.
Please verify:
- All timestamps are in valid formats
- start_time < end_time
- No overlapping segments
- Timestamps are within media duration
Return issues in Markdown format.
"""
    sample = df.sample(min(sample_size, len(df))).to_dict(orient="records")
    return ask_openai(prompt, sample)

# 4. Bounding Box Consistency
def check_bboxes_with_llm(df, sample_size=10):
    prompt = """
You are reviewing a dataset with image annotations using bounding boxes.
Please check for:
- Boxes that exceed image bounds
- Inconsistent label usage for similar regions
- Suspiciously small or large boxes
- Duplicates or overlaps
Return issues and improvement tips in Markdown format.
"""
    sample = df.sample(min(sample_size, len(df))).to_dict(orient="records")
    return ask_openai(prompt, sample)

# 5. Master QA runner
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
