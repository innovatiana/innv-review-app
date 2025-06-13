import pandas as pd
import openai
import streamlit as st

openai.api_key = st.secrets["openai"]["api_key"]

def check_ner_conflicts_with_llm(df):
    sample = df.sample(min(10, len(df))).to_dict(orient='records')

    prompt = f"""
You are a data validation expert reviewing NER annotations.
Each item is a text annotated with one or more named entities. 
Please analyze the data below and list:
- Any inconsistencies or conflicts in entity spans
- Overlapping spans with different labels
- Suspicious patterns that look like annotation errors

Dataset sample:

{sample}

Return the result in Markdown bullet points.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]

def check_prompt_response_alignment_llm(df):
    sample = df.sample(min(10, len(df))).to_dict(orient='records')

    prompt = f"""
You are reviewing a dataset of prompt-response pairs intended for training large language models.
Please analyze the data and identify any of the following:
- Responses that are too short, too long, or empty
- Prompts and responses that are nearly identical
- Any signs of generic or templated answers

Dataset sample:

{sample}

Return the result in Markdown bullet points.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]

def check_timestamp_validity_llm(df):
    sample = df.sample(min(10, len(df))).to_dict(orient='records')

    prompt = f"""
You are a QA assistant reviewing a dataset containing timestamped segments (audio or video).
Please analyze the following:
- Are all timestamps in a valid format?
- Are start times always before end times?
- Are there any overlaps or timestamp inconsistencies?

Dataset sample:

{sample}

Return the result in Markdown bullet points.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]

def check_bounding_box_consistency_llm(df):
    sample = df.sample(min(10, len(df))).to_dict(orient='records')

    prompt = f"""
You are reviewing a dataset of image annotations using bounding boxes.
Please look for the following:
- Boxes with invalid coordinates (e.g., negative values or exceeding image size)
- Overlapping or duplicated boxes
- Labels that do not match the object in the image filename (if applicable)

Dataset sample:

{sample}

Return the result in Markdown bullet points.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]
