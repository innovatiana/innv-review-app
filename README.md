# 🧠 AI Dataset QA Review Tool

This small app helps **review and validate datasets annotated for AI training**, across multiple modalities and formats. It performs **automatic quality checks** and highlights potential issues in the data, including:

- Text classification and NER datasets
- LLM fine-tuning (prompt/response) datasets
- Object detection datasets (bounding boxes, COCO-style)
- Audio/video timestamp-based annotations
- Multimodal metadata (JSON, JSONL, XML, CSV)

---

## 🚀 Features

### ✅ General Quality Checks
- Missing values
- Duplicate rows
- Label distribution and class imbalance

### 🧠 Machine Learning Checks
- Cleanlab-based anomaly detection
- Model-based prediction vs label mismatch

### ✍️ NLP & NER Checks
- Entity span overlap detection
- Conflicting labels on same span
- Text length vs entity density
- Language mismatch (via langdetect)

### 🤖 LLM Prompt/Response QA
- Missing prompt or response
- Long/short responses
- Semantic similarity scoring (via sentence-transformers)

### 🕒 Timestamp QA
- Format validation (ISO 8601, hh:mm:ss, float)
- `start_time < end_time`
- Overlapping segments
- Timestamps beyond media duration (coming soon)

### 🖼️ Object Detection QA
- Invalid bounding boxes
- Bboxes outside image boundaries

---

## 📂 Supported Formats

You can upload:
- `.csv`
- `.json`
- `.jsonl`
- `.xml`
- `.zip` (with supported metadata files inside)

---

## 📦 Installation

```bash
git clone https://github.com/your-org/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run main.py
