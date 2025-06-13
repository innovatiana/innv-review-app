# app.py - QA Review Tool with Cleanlab scoring, reviewer view, and comments
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="QA Review Manager", layout="wide")
st.title("🔎 QA Review Manager with Cleanlab Scoring")

st.markdown("""
Ce tableau vous permet d'analyser un dataset annoté pour identifier les échantillons qui semblent incohérents ou mal étiquetés.

- 🧠 L'outil utilise [Cleanlab](https://github.com/cleanlab/cleanlab) pour détecter les erreurs probables à l'aide d'un modèle automatique.
- 📊 Chaque ligne reçoit un **score d'erreur** et une **priorité de revue**.
- 👤 Vous pouvez ensuite **assigner des revues** aux annotateurs QA et suivre leur progression.
""")

# Simulated task store
if 'tasks' not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=['index', 'label', 'error_score', 'review_priority', 'assigned_to', 'status', 'comment'])

# Cleanlab scoring logic
@st.cache_data(show_spinner=True)
def score_with_cleanlab(df):
    if 'label' not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'label'.")

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
    error_score[np.array(issues, dtype=int)] = 1

    df_out = df.copy()
    df_out['error_score'] = error_score
    df_out['review_priority'] = pd.qcut(
        error_score + np.random.rand(len(df)) * 0.01,
        q=4,
        labels=["Faible", "Moyenne", "Élevée", "Critique"]
    )
    return df_out

# Upload dataset
uploaded_file = st.file_uploader("📂 Importer un fichier CSV annoté (avec colonne 'label')", type="csv")

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = score_with_cleanlab(df_raw)

        st.success("✅ Analyse terminée. Voici vos données annotées et scorées.")
        with st.expander("Afficher les données analysées"):
            st.dataframe(df[['label', 'error_score', 'review_priority']].sort_values(by='error_score', ascending=False), use_container_width=True)

        st.subheader("📌 Assigner des revues QA")
        st.markdown("Choisissez des lignes à faire relire par un annotateur qualité.")

        to_assign = st.multiselect("Sélectionnez les entrées à assigner :", options=df.index.tolist())
        reviewer = st.text_input("Nom ou email du relecteur :")
        if st.button("✅ Assigner") and reviewer and to_assign:
            new_tasks = df.loc[to_assign][['label', 'error_score', 'review_priority']].copy()
            new_tasks['index'] = new_tasks.index
            new_tasks['assigned_to'] = reviewer
            new_tasks['status'] = 'À revoir'
            new_tasks['comment'] = ""
            st.session_state.tasks = pd.concat([st.session_state.tasks, new_tasks], ignore_index=True)
            st.success(f"🎯 {len(to_assign)} entrées assignées à {reviewer}.")

        st.subheader("📋 Liste des revues en cours")
        status_filter = st.selectbox("Filtrer par statut :", ["Toutes", "À revoir", "Confirmée", "Corrigée", "Rejetée"])
        tasks = st.session_state.tasks.copy()
        if status_filter != "Toutes":
            tasks = tasks[tasks['status'] == status_filter]
        st.dataframe(tasks, use_container_width=True)

        st.subheader("✏️ Mettre à jour une tâche de revue")
        if not tasks.empty:
            task_ids = tasks['index'].tolist()
            task_to_update = st.selectbox("ID de tâche à mettre à jour :", options=task_ids)
            new_status = st.selectbox("Nouveau statut :", ["Confirmée", "Corrigée", "Rejetée"])
            new_comment = st.text_area("Commentaire (optionnel) :")
            if st.button("🔄 Mettre à jour le statut"):
                st.session_state.tasks.loc[st.session_state.tasks['index'] == task_to_update, 'status'] = new_status
                st.session_state.tasks.loc[st.session_state.tasks['index'] == task_to_update, 'comment'] = new_comment
                st.success(f"✅ Tâche {task_to_update} mise à jour en '{new_status}'.")

        st.subheader("👤 Vue annotateur")
        annotator_filter = st.text_input("Nom ou email annotateur :")
        if annotator_filter:
            annot_tasks = st.session_state.tasks[st.session_state.tasks['assigned_to'] == annotator_filter]
            st.markdown(f"Tâches assignées à **{annotator_filter}**")
            st.dataframe(annot_tasks, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du dataset : {e}")
else:
    st.info("Importez un fichier CSV avec des colonnes de features + une colonne 'label' pour commencer l’analyse.")
