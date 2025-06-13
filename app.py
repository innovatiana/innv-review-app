# app.py - QA Review Tool with Full Quality Controls

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import xml.etree.ElementTree as ET
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from cleanlab.classification import CleanLearning
from collections import Counter
from langdetect import detect
from datetime import datetime
from dateutil import parser as date_parser
from PIL import Image
import urllib.request
import io

st.set_page_config(page_title="QA Review Tool - Full QC", layout="wide")
st.title("🔍 Outil de Revue Qualité des Datasets Annotés")

st.markdown("""
Cet outil vous permet d'importer différents types de fichiers et d'effectuer automatiquement des **contrôles qualité** selon la modalité (texte, image, NER, LLM, etc.).
""")

uploaded_file = st.sidebar.file_uploader("📂 Importer un fichier annoté", type=["csv", "json", "xml", "jsonl"])
dataset_type = st.sidebar.selectbox("Type de données", ["CSV (tabulaire)", "JSON (NER)", "JSONL (LLM)", "XML", "Multimodal (COCO/XML)"])

if uploaded_file:
    try:
        st.header("🧪 Analyse Automatique du Dataset")

        if dataset_type == "CSV (tabulaire)":
            df = pd.read_csv(uploaded_file)

            st.subheader("📊 Distribution des labels")
            if 'label' in df.columns:
                fig = px.histogram(df, x='label', title="Distribution des classes")
                st.plotly_chart(fig)

            st.subheader("📌 Doublons")
            duplicates = df[df.duplicated()]
            st.write(f"Nombre de doublons : {len(duplicates)}")
            if not duplicates.empty:
                st.dataframe(duplicates)

            st.subheader("⚠️ Valeurs manquantes")
            st.write(df.isnull().sum())

            st.subheader("📈 Corrélation entre features")
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                fig, ax = plt.subplots()
                sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            st.subheader("🧠 Cleanlab - Détection d'anomalies")
            if 'label' in df.columns:
                try:
                    X = df.drop(columns=['label'])
                    X = pd.get_dummies(X)
                    if X.isnull().values.any():
                        X = X.fillna(0)

                    le = LabelEncoder()
                    y = le.fit_transform(df['label'])

                    clean_model = CleanLearning(clf=RandomForestClassifier(), cv_n_folds=2)
                    clean_model.fit(X.values, y)
                    issues = clean_model.get_label_issues()
                    df['error_score'] = 0
                    df.loc[issues, 'error_score'] = 1

                    st.success(f"✅ {len(issues)} anomalies détectées avec Cleanlab")
                    st.dataframe(df[df['error_score'] == 1])
                except Exception as e:
                    st.error(f"Erreur Cleanlab : {e}")

        elif dataset_type == "JSON (NER)":
            data = json.load(uploaded_file)
            if isinstance(data, dict):
                data = [data]
            for i, entry in enumerate(data):
                text = entry.get("text", "")
                entities = entry.get("entities", [])
                warnings = []
                st.markdown(f"**Texte #{i+1}**: {text}")
                for ent in entities:
                    start, end, label = ent['start'], ent['end'], ent['label']
                    span = text[start:end]
                    if span.strip() == "":
                        warnings.append(f"Entité vide entre {start}-{end}")
                    if not re.match(r'^\w+', span):
                        warnings.append(f"Entité suspecte : '{span}'")
                token_counts = Counter(text.split())
                st.write("Top tokens:", token_counts.most_common(5))
                try:
                    lang = detect(text)
                    st.write("Langue détectée:", lang)
                except:
                    st.warning("Langue non détectable")
                if warnings:
                    st.warning("\n".join(warnings))

        elif dataset_type == "JSONL (LLM)":
            lines = uploaded_file.readlines()
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            for i, line in enumerate(lines):
                data = json.loads(line.decode("utf-8"))
                prompt, response = data.get("prompt"), data.get("response")
                if prompt and response:
                    st.markdown(f"**Prompt:** {prompt}")
                    st.markdown(f"**Réponse:** {response}")
                    if len(response.split()) < 3:
                        st.warning("Réponse trop courte")
                    emb1 = model.encode(prompt, convert_to_tensor=True)
                    emb2 = model.encode(response, convert_to_tensor=True)
                    sim = float(util.cos_sim(emb1, emb2))
                    st.write(f"Similarité sémantique : {sim:.2f}")
                else:
                    st.error("Prompt ou réponse manquant(e)")

        elif dataset_type == "XML":
            content = uploaded_file.read().decode("utf-8")
            root = ET.fromstring(content)
            entries = root.findall(".//entry")
            st.write(f"Entrées détectées : {len(entries)}")
            for entry in entries[:5]:
                st.text(ET.tostring(entry, encoding='unicode'))

        elif dataset_type == "Multimodal (COCO/XML)":
            st.info("Analyse COCO / multimédia basique en cours...")
            content = json.load(uploaded_file)
            if 'images' in content and 'annotations' in content:
                paths = [img.get('file_name') for img in content['images']]
                st.write(f"Images référencées : {len(paths)}")
                missing = [p for p in paths if not os.path.exists(p)]
                if missing:
                    st.error(f"{len(missing)} fichiers manquants")
                annots_by_image = Counter([a['image_id'] for a in content['annotations']])
                st.bar_chart(pd.Series(annots_by_image))

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Veuillez importer un fichier pour démarrer l'analyse.")
