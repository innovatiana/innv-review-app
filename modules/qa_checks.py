import os
import pandas as pd
from openai import OpenAI
import streamlit as st

# Initialize OpenAI client securely
client = OpenAI(api_key="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")

def run_all_quality_checks(dataset: pd.DataFrame, metadata: dict = None):
    results = {}

    # Example: Simple null check
    try:
        null_counts = dataset.isnull().sum().to_dict()
        results["null_values"] = null_counts
    except Exception as e:
        results["null_values"] = f"Error: {str(e)}"

    # Example: LLM validation (just a basic call)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a data quality assistant."},
                {"role": "user", "content": "Check this dataset for suspicious patterns: " + dataset.head(3).to_json()}
            ]
        )
        results["llm_feedback"] = response.choices[0].message.content
    except Exception as e:
        results["llm_feedback"] = f"OpenAI Error: {str(e)}"

    return results
