# modules/qa_checks.py
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from collections import Counter
import langdetect
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def is_valid_timestamp(ts):
    try:
        if isinstance(ts, (int, float)):
            return True
        ts_str = str(ts)
        iso_regex = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        time_regex = r"^\d{1,2}:\d{2}(:\d{2})?$"
        float_regex = r"^\d+(\.\d+)?$"
        return re.match(iso_regex, ts_str) or re.match(time_regex, ts_str) or re.match(float_regex, ts_str)
    except:
        return False

def detect_ner_span_conflicts(df):
    if not {"start", "end", "label"}.issubset(df.columns):
        return None

    overlap_count = 0
    conflict_count = 0

    for text_id, group in df.groupby("text_id") if "text_id" in df.columns else [(None, df)]:
        spans = group[["start", "end", "label"]].sort_values("start").values
        for i in range(len(spans) - 1):
            s1, e1, l1 = spans[i]
            s2, e2, l2 = spans[i + 1]
            if s2 < e1:
                overlap_count += 1
                if l1 != l2:
                    conflict_count += 1

    return {"span_overlaps": overlap_count, "conflicting_labels_on_span": conflict_count}

def llm_prompt_response_validation(df):
    issues = {
        "missing_prompt": 0,
        "missing_response": 0,
        "long_response": 0,
        "short_response": 0,
        "low_similarity": 0
    }
    try:
        if {"prompt", "response"}.issubset(df.columns):
            for _, row in df.iterrows():
                prompt = str(row["prompt"])
                response = str(row["response"])
                if not prompt.strip():
                    issues["missing_prompt"] += 1
                if not response.strip():
                    issues["missing_response"] += 1
                if len(response) > 2000:
                    issues["long_response"] += 1
                if len(response) < 5:
                    issues["short_response"] += 1

                emb1 = model.encode(prompt, convert_to_tensor=True)
                emb2 = model.encode(response, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(emb1, emb2).item()
                if sim < 0.3:
                    issues["low_similarity"] += 1
        return issues
    except:
        return issues

def run_all_quality_checks(df, metadata):
    report = {}
    report["num_rows"] = len(df)
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_rows"] = df.duplicated().sum()

    if "text" in df.columns:
        token_counts = Counter(" ".join(df["text"].astype(str)).split())
        report["token_frequency"] = dict(token_counts.most_common(20))
        report["avg_text_length"] = df["text"].astype(str).apply(len).mean()

        try:
            detected_languages = df["text"].astype(str).apply(langdetect.detect)
            report["language_distribution"] = dict(Counter(detected_languages))
        except:
            report["language_distribution"] = "Error detecting language"

    if "label" in df.columns:
        try:
            df = df.dropna(subset=["text", "label"])
            le = LabelEncoder()
            y = le.fit_transform(df["label"])
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000)
            X_vec = vectorizer.fit_transform(df["text"].astype(str))
            model_lr = LogisticRegression(max_iter=1000)
            clf = CleanLearning(clf=model_lr)
            clf.fit(X_vec, y)
            label_issues = clf.get_label_issues()
            report["label_issues"] = int(label_issues.sum())
        except Exception as e:
            report["label_issues"] = f"Error: {e}"

    if {"bbox_x", "bbox_y", "bbox_width", "bbox_height"}.issubset(df.columns):
        invalid_bboxes = df[
            (df["bbox_width"] <= 0) |
            (df["bbox_height"] <= 0) |
            (df["bbox_x"] < 0) |
            (df["bbox_y"] < 0)
        ]
        report["invalid_bounding_boxes"] = len(invalid_bboxes)

        if {"image_width", "image_height"}.issubset(df.columns):
            outside_bounds = df[
                (df["bbox_x"] + df["bbox_width"] > df["image_width"]) |
                (df["bbox_y"] + df["bbox_height"] > df["image_height"])
            ]
            report["bounding_boxes_outside_image"] = len(outside_bounds)

    if {"start_time", "end_time"}.issubset(df.columns):
        invalid_time_order = df[df["start_time"] >= df["end_time"]]
        report["invalid_timestamp_order"] = len(invalid_time_order)

        invalid_formats = df[~df["start_time"].apply(is_valid_timestamp) | ~df["end_time"].apply(is_valid_timestamp)]
        report["invalid_timestamp_format"] = len(invalid_formats)

        if "media_id" in df.columns:
            overlap_count = 0
            for media_id, group in df.groupby("media_id"):
                sorted_group = group.sort_values("start_time")
                ends = sorted_group["end_time"].tolist()
                starts = sorted_group["start_time"].tolist()[1:]
                prev_ends = ends[:-1]
                for s, e in zip(starts, prev_ends):
                    if s < e:
                        overlap_count += 1
            report["overlapping_segments"] = overlap_count

    ner_conflicts = detect_ner_span_conflicts(df)
    if ner_conflicts:
        report.update(ner_conflicts)

    llm_checks = llm_prompt_response_validation(df)
    report.update(llm_checks)

    return report
