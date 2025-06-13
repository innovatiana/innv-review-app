# modules/qa_checks.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import langdetect

# Cleanlab (optional check)
try:
    from cleanlab.classification import CleanLearning
    from cleanlab.dataset import cleanlab_scores
    has_cleanlab = True
except ImportError:
    has_cleanlab = False

def check_class_distribution(df):
    if 'label' in df.columns:
        return df['label'].value_counts().to_frame(name='count')
    return "⚠️ No 'label' column found."

def check_missing_values(df):
    missing = df.isnull().sum()
    return missing[missing > 0].to_frame(name='missing_count') if not missing.empty else "✅ No missing values."

def check_duplicates(df):
    duplicates = df[df.duplicated(keep=False)]
    return duplicates if not duplicates.empty else "✅ No duplicates."

def check_token_length(df):
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).apply(len)
        return df[['text', 'text_length']].sort_values(by='text_length', ascending=False).head(10)
    return "⚠️ No 'text' column to check."

def check_language(df):
    if 'text' in df.columns:
        try:
            df['lang'] = df['text'].astype(str).apply(lambda x: langdetect.detect(x) if len(x) > 5 else 'unknown')
            return df[['text', 'lang']].head(10)
        except:
            return "⚠️ Language detection failed."
    return "⚠️ No 'text' column."

def check_prompt_response(df):
    if 'prompt' in df.columns and 'response' in df.columns:
        empty_prompt = df['prompt'].isnull().sum()
        empty_resp = df['response'].isnull().sum()
        long_resp = df['response'].astype(str).apply(len).gt(1000).sum()
        return {
            "Empty prompts": empty_prompt,
            "Empty responses": empty_resp,
            "Long responses >1000 chars": long_resp
        }
    return "⚠️ Columns 'prompt' and 'response' not found."

def check_span_conflicts(df):
    if {'start', 'end', 'label'}.issubset(df.columns):
        overlaps = []
        for i, row in df.iterrows():
            for j, other in df.iterrows():
                if i != j and row['start'] < other['end'] and row['end'] > other['start']:
                    overlaps.append((i, j))
        return pd.DataFrame(overlaps, columns=['Entity_1', 'Entity_2']) if overlaps else "✅ No span conflicts."
    return "⚠️ Columns 'start', 'end', 'label' required."

def check_bbox_consistency(df):
    if {'x', 'y', 'width', 'height', 'image_width', 'image_height'}.issubset(df.columns):
        df['bbox_valid'] = (
            (df['x'] >= 0) & (df['y'] >= 0) &
            (df['x'] + df['width'] <= df['image_width']) &
            (df['y'] + df['height'] <= df['image_height'])
        )
        return df[df['bbox_valid'] == False] if df['bbox_valid'].eq(False).any() else "✅ All bounding boxes are within image dimensions."
    return "⚠️ Required columns missing for bbox validation."

def check_timestamp_validity(df):
    if {'start_time', 'end_time'}.issubset(df.columns):
        invalid = df[df['start_time'] > df['end_time']]
        return invalid if not invalid.empty else "✅ All timestamps are valid."
    return "⚠️ Columns 'start_time' and 'end_time' required."

def check_cleanlab(df):
    if not has_cleanlab:
        return "⚠️ Cleanlab not installed."
    if 'text' not in df.columns or 'label' not in df.columns:
        return "⚠️ 'text' and 'label' required for Cleanlab."
    try:
        X = df['text'].astype(str)
        y = df['label']
        vec = lambda x: np.array([hash(w) % 1000 for w in x.lower().split()])[:10]
        X_vect = np.vstack(X.map(vec))
        X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred_probs = model.predict_proba(X_vect)
        cl = CleanLearning(model)
        ranked = cleanlab_scores(cl.clf(), X_vect, y)
        outliers = pd.Series(ranked).sort_values(ascending=False).head(10)
        return outliers.to_frame(name='cleanlab_score')
    except Exception as e:
        return f"❌ Cleanlab error: {e}"


# Mapping all checks to their function
check_functions = {
    "📊 Class imbalance / label distribution": check_class_distribution,
    "🧼 Missing values": check_missing_values,
    "🧾 Duplicates": check_duplicates,
    "🗣️ Token frequency & text length": check_token_length,
    "🌍 Language detection": check_language,
    "🤖 Prompt/Response validation (LLM)": check_prompt_response,
    "🧵 NER span overlap / conflict": check_span_conflicts,
    "🖼️ Bounding box consistency": check_bbox_consistency,
    "🕒 Timestamp validation": check_timestamp_validity,
    "🧠 Cleanlab anomaly detection (classification)": check_cleanlab
}

def run_all_quality_checks(df, metadata):
    results = {}
    for check in metadata.get("selected_checks", []):
        func = check_functions.get(check)
        if func:
            try:
                results[check] = func(df)
            except Exception as e:
                results[check] = f"❌ Error running {check}: {e}"
    return results
