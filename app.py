# app.py - QA Review Tool with Cleanlab-based scoring and Streamlit UI
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cleanlab.classification import CleanLearning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="QA Review Manager", layout="wide")
st.title("🔎 Innovatiana's QA Review Manager with Cleanlab Scoring")

# Simulated task store
if 'tasks' not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=['index', 'label', 'error_score', 'review_priority', 'assigned_to', 'status'])

# Cleanlab scoring logic
@st.cache_data(show_spinner=True)
def score_with_cleanlab(df):
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = df.drop(columns=['label'])
    y = df['label']

    if X.select_dtypes(include='object').shape[1] > 0:
        X = pd.get_dummies(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = CleanLearning(RandomForestClassifier(random_state=42), cv_n_folds=2)
    clf.fit(X.values, y_enc)
    issues = clf.get_label_issues()
    error_score = np.zeros(len(df))
    error_score[issues] = 1  # binary error flag for simplicity

    df_out = df.copy()
    df_out['error_score'] = error_score
    df_out['review_priority'] = pd.qcut(error_score + np.random.rand(len(df)) * 0.01, q=4, labels=["Low", "Medium", "High", "Critical"])
    return df_out

# Upload dataset
uploaded_file = st.file_uploader("Upload annotated dataset (CSV with label column)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    try:
        df = score_with_cleanlab(df_raw)
        st.subheader("📋 Scored Entries")
        st.dataframe(df[['label', 'error_score', 'review_priority']].sort_values(by='error_score', ascending=False), use_container_width=True)

        st.subheader("🛠️ Assign Review Tasks")
        to_assign = st.multiselect("Select entries to assign:", options=df.index.tolist())
        reviewer = st.text_input("Assign to reviewer (name or email):")
        if st.button("Assign") and reviewer and to_assign:
            new_tasks = df.loc[to_assign][['label', 'error_score', 'review_priority']].copy()
            new_tasks['index'] = new_tasks.index
            new_tasks['assigned_to'] = reviewer
            new_tasks['status'] = 'To Review'
            st.session_state.tasks = pd.concat([st.session_state.tasks, new_tasks], ignore_index=True)
            st.success(f"Assigned {len(to_assign)} entries to {reviewer}.")

        st.subheader("✅ Review To-Do List")
        status_filter = st.selectbox("Filter by status:", ["All", "To Review", "Confirmed", "Corrected", "Rejected"])
        tasks = st.session_state.tasks.copy()
        if status_filter != "All":
            tasks = tasks[tasks['status'] == status_filter]
        st.dataframe(tasks, use_container_width=True)

        st.subheader("✏️ Update Task Status")
        if not tasks.empty:
            task_ids = tasks['index'].tolist()
            task_to_update = st.selectbox("Select task ID:", options=task_ids)
            new_status = st.selectbox("New status:", ["Confirmed", "Corrected", "Rejected"])
            if st.button("Update Status"):
                st.session_state.tasks.loc[st.session_state.tasks['index'] == task_to_update, 'status'] = new_status
                st.success(f"Task {task_to_update} updated to {new_status}.")
    except Exception as e:
        st.error(f"❌ Error while processing dataset: {e}")
else:
    st.info("Please upload a CSV file containing at least a 'label' column.")
