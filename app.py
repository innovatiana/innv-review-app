# app.py - QA Review Tool for NER JSON + Cleanlab scoring for tabular
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from sklearn.ensemble import RandomForestClassifier
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="QA Review Manager (Text & NER)", layout="wide")
st.title("🔎 QA Review Manager – Cleanlab + NER JSON")

st.markdown("""
Cet outil vous permet :

- D'importer un fichier CSV annoté (classification tabulaire) OU un JSON de type NER
- D'obtenir un scoring automatique des erreurs via Cleanlab (CSV)
- De revoir des entités annotées dans un texte (NER) avec une interface colorée
- D’identifier les doublons ou entités incorrectes (NER)
""")

if 'tasks' not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=['index', 'label', 'error_score', 'review_priority', 'assigned_to', 'status', 'comment'])

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
    df_out['review_priority'] = pd.qcut(error_score + np.random.rand(len(df)) * 0.01, q=4, labels=["Faible", "Moyenne", "Élevée", "Critique"])
    return df_out

st.sidebar.header("📂 Importer vos données")
dataset_type = st.sidebar.radio("Type de fichier", ["CSV (tabulaire)", "JSON (NER)"])

uploaded_file = st.sidebar.file_uploader("Fichier à importer", type=["csv", "json"])

if uploaded_file:
    try:
        if dataset_type == "CSV (tabulaire)":
            df_raw = pd.read_csv(uploaded_file)
            df = score_with_cleanlab(df_raw)
            st.success("✅ Analyse terminée. Voici vos données annotées et scorées.")
            with st.expander("Afficher les données analysées"):
                st.dataframe(df[['label', 'error_score', 'review_priority']].sort_values(by='error_score', ascending=False), use_container_width=True)

        elif dataset_type == "JSON (NER)":
            data = json.load(uploaded_file)
            if isinstance(data, dict):
                data = [data]
            st.success("✅ JSON chargé avec succès. Revue des entités annotées :")

            for i, entry in enumerate(data):
                st.markdown(f"### Exemple {i+1}")
                text = entry.get("text", "")
                entities = entry.get("entities", [])

                # Validation heuristique : doublons + mauvais alignements
                entity_set = set()
                warnings = []
                colored = ""
                last_idx = 0

                for ent in sorted(entities, key=lambda x: x['start']):
                    start, end, label = ent['start'], ent['end'], ent['label']
                    span = text[start:end]

                    if (start, end, label) in entity_set:
                        warnings.append(f"🔁 Doublon : {span} ({label})")
                    else:
                        entity_set.add((start, end, label))

                    if not span.strip():
                        warnings.append(f"⚠️ Entité vide ou blanche aux positions {start}-{end}")

                    if not re.match(r'^\w+', span):
                        warnings.append(f"❓ Entité suspecte : '{span}'")

                    colored += text[last_idx:start] + f"<span style='background-color: #ffd'>{span} <sub>[{label}]</sub></span>"
                    last_idx = end

                colored += text[last_idx:]
                st.markdown(colored, unsafe_allow_html=True)

                if warnings:
                    st.warning("\n".join(warnings))
                else:
                    st.success("Aucune anomalie détectée.")

    except Exception as e:
        st.error(f"❌ Erreur lors de l'import : {e}")
else:
    st.info("Importez un fichier pour démarrer. Choisissez le bon format à gauche.")
