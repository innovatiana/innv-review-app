# app.py - QA Review Tool for multiple formats: CSV, JSON, JSONL, XML, COCO, LLM
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="QA Review Manager", layout="wide")
st.title("🔎 QA Review Manager – Multi-format QA Review")

st.markdown("""
Cet outil vous permet de :

- Importer différents formats de fichiers (CSV, JSON, JSONL, XML, COCO, etc.)
- Revoir visuellement et valider des annotations NER, transcription, image, audio, vidéo ou texte génératif (prompt/réponse)
- Collaborer efficacement : assigner des tâches, suivre les statuts, commenter
- Détecter automatiquement des erreurs avec heuristiques et Cleanlab
""")

if 'tasks' not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=['index', 'source_type', 'content', 'assigned_to', 'status', 'comment'])

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
dataset_type = st.sidebar.radio("Type de fichier", ["CSV (tabulaire)", "JSON (NER)", "JSONL", "XML", "JSON/XML/COCO multimédia", "Prompt/Réponse LLM"])

uploaded_file = st.sidebar.file_uploader("Fichier à importer", type=["csv", "json", "jsonl", "xml", "txt"])

if uploaded_file:
    try:
        if dataset_type == "CSV (tabulaire)":
            df_raw = pd.read_csv(uploaded_file)
            df = score_with_cleanlab(df_raw)
            st.success("✅ Analyse terminée. Voici vos données annotées et scorées.")
            st.dataframe(df[['label', 'error_score', 'review_priority']], use_container_width=True)

        elif dataset_type == "JSON (NER)":
            data = json.load(uploaded_file)
            if isinstance(data, dict):
                data = [data]
            st.success("✅ JSON chargé avec succès. Revue des entités annotées :")
            for i, entry in enumerate(data):
                text = entry.get("text", "")
                entities = entry.get("entities", [])
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
                st.session_state.tasks = pd.concat([st.session_state.tasks, pd.DataFrame([{
                    'index': i,
                    'source_type': 'NER',
                    'content': text,
                    'assigned_to': '',
                    'status': 'À revoir',
                    'comment': ''
                }])], ignore_index=True)

        elif dataset_type == "JSONL":
            lines = uploaded_file.readlines()
            for i, line in enumerate(lines):
                data = json.loads(line.decode("utf-8"))
                text = json.dumps(data, indent=2)
                st.text_area(f"Entrée {i+1}", text, height=150)
                st.session_state.tasks = pd.concat([st.session_state.tasks, pd.DataFrame([{
                    'index': i,
                    'source_type': 'JSONL',
                    'content': text,
                    'assigned_to': '',
                    'status': 'À revoir',
                    'comment': ''
                }])], ignore_index=True)

        elif dataset_type == "XML":
            content = uploaded_file.read().decode("utf-8")
            root = ET.fromstring(content)
            entries = root.findall(".//entry")
            for i, entry in enumerate(entries):
                st.text_area(f"Entrée XML {i+1}", ET.tostring(entry, encoding='unicode'), height=100)
                st.session_state.tasks = pd.concat([st.session_state.tasks, pd.DataFrame([{
                    'index': i,
                    'source_type': 'XML',
                    'content': ET.tostring(entry, encoding='unicode'),
                    'assigned_to': '',
                    'status': 'À revoir',
                    'comment': ''
                }])], ignore_index=True)

        elif dataset_type == "Prompt/Réponse LLM":
            raw_text = uploaded_file.read().decode("utf-8")
            lines = raw_text.strip().split('\n')
            for i, line in enumerate(lines):
                if '\t' in line:
                    prompt, response = line.split('\t')
                    st.markdown(f"**Prompt**: {prompt}")
                    st.markdown(f"**Réponse**: {response}")
                    st.session_state.tasks = pd.concat([st.session_state.tasks, pd.DataFrame([{
                        'index': i,
                        'source_type': 'LLM',
                        'content': prompt + "\n---\n" + response,
                        'assigned_to': '',
                        'status': 'À revoir',
                        'comment': ''
                    }])], ignore_index=True)

        elif dataset_type == "JSON/XML/COCO multimédia":
            st.info("Module multimédia à implémenter : support image/audio/vidéo en cours. Veuillez charger un JSON avec chemins de fichiers et labels associés.")

        # Interface collaborative commune
        st.subheader("📌 Assigner des tâches à revoir")
        ids = st.session_state.tasks.index.tolist()
        selected_ids = st.multiselect("Sélectionnez des tâches à assigner :", ids)
        reviewer = st.text_input("Nom ou email annotateur :")
        if st.button("✅ Assigner") and reviewer:
            st.session_state.tasks.loc[selected_ids, 'assigned_to'] = reviewer
            st.success(f"Tâches assignées à {reviewer}.")

        st.subheader("📋 Liste collaborative de revues")
        st.dataframe(st.session_state.tasks, use_container_width=True)

        st.subheader("✏️ Mettre à jour une tâche")
        task_to_update = st.selectbox("ID de tâche à modifier :", st.session_state.tasks['index'].tolist())
        new_status = st.selectbox("Nouveau statut :", ["À revoir", "Corrigée", "Confirmée", "Rejetée"])
        comment = st.text_area("Commentaire")
        if st.button("🔄 Mettre à jour"):
            st.session_state.tasks.loc[st.session_state.tasks['index'] == task_to_update, 'status'] = new_status
            st.session_state.tasks.loc[st.session_state.tasks['index'] == task_to_update, 'comment'] = comment
            st.success("Tâche mise à jour.")

        st.subheader("👤 Vue annotateur")
        name_filter = st.text_input("Filtrer par annotateur")
        if name_filter:
            st.dataframe(st.session_state.tasks[st.session_state.tasks['assigned_to'] == name_filter], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier : {e}")
else:
    st.info("Importez un fichier compatible et commencez la revue QA collaborative.")
