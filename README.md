# AutoQAlytics 
_Transforming Quality Assurance with Intelligent Automation_

AutoQAlytics is an end-to-end NLP-driven framework for automating software quality assurance. From ingesting bug-report data to classification, entity extraction, summarization, clustering, and a unified REST API—AutoQAlytics accelerates triage, surfaces insights, and drives smarter QA decisions.

---  

## Table of Contents  
1. [Features](#features)  
2. [Architecture Overview](#architecture-overview)  
3. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Environment Setup](#environment-setup)  
4. [Data Acquisition](#data-acquisition)  
5. [ETL Pipeline](#etl-pipeline)  
6. [Dataset Overview](#dataset-overview)  
7. [Classification](#classification)  
   - [TF-IDF + Logistic Regression](#tf-idf--logistic-regression)  
   - [Transformer (BERT/RoBERTa) Fine-Tuning](#transformer-bertroberta-fine-tuning)  
8. [Named Entity Recognition (NER)](#named-entity-recognition-ner)  
9. [Summarization](#summarization)  
10. [Clustering & Trend Analysis](#clustering--trend-analysis)  
11. [REST API Integration](#rest-api-integration)  
12. [Testing the API](#testing-the-api)  
13. [Repository Structure](#repository-structure)  

---

## Features  
- **Data Ingestion**: Fetch GitHub issues (open & closed) with full metadata.  
- **ETL**: Clean, dedupe, tokenize, lemmatize, and enrich records with severity & category labels.  
- **Exploratory Overview**: Quick stats and crosstabs of your issue corpus.  
- **Classification**:  
  - Lightweight TF-IDF + Logistic Regression  
  - High-accuracy Transformer (BERT/RoBERTa) fine-tuning  
- **NER**: Hybrid rule-based + trainable spaCy pipeline for extracting error codes, modules, OS, etc.  
- **Summarization**:  
  - Extractive (TextRank)  
  - Abstractive (BART/T5)  
- **Clustering**: Sentence-Transformer embeddings → UMAP visualization → KMeans/HDBSCAN grouping  
- **REST API**: FastAPI service exposing all pipelines (classify, ner, summarize, cluster)  
- **Testing**: Automated test script to validate your local server  

---

## Architecture Overview 

[ GitHub Issues ] ↓ fetch_issues.py [ Raw JSON Data ] ↓ etl_pipeline.py [ Cleaned JSONL ] ↓ dataset_overview.py ← sanity checks ↓ train_tfidf_classifier.py ← severity/category ↓ train_transformer_classifier.py ↓ ner_pipeline.py ← spaCy NER model ↓ summarization_pipeline.py ← summaries ↓ clustering_pipeline.py ← clusters & UMAP ↓ main.py (FastAPI) ← unified API ↓ test_api.py ← integration tests

---

## Getting Started  

### Prerequisites  
- Python 3.8+  
- GitHub Personal Access Token with `repo:issues` scope  
- (Optional) CUDA-enabled GPU for Transformer fine-tuning  

### Environment Setup  
1. Clone this repo:  
    ```bash
    git clone https://github.com/antonshumov2023/AutoQAlytics.git
    cd AutoQAlytics
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Export your GitHub token:
    ```bash
    export GITHUB_TOKEN="ghp_your_token_here"
    ```

---

## Data Acquisition
Fetch all issues from target repo:

    ```bash
    python fetch_issues.py --owner ownername --repo reponame --output owner_repo_issues.json
    ```

---

## ETL Pipeline
Clean, dedupe, tokenize, map labels, compute resolution days:

    ```bash
    python etl_pipeline.py --input owner_repo_issues.json --output issues_dataset.jsonl
    ```

---

## Dataset Overview
Quick stats on a new dataset:

    ```bash
    python dataset_overview.py --input issues_dataset.jsonl
    ```

---

## Classification
TF-IDF + Logistic Regression

    ```bash
    python train_tfidf_classifier.py --input issues_dataset.jsonl --target severity --output model_tfidf_severity.joblib
    python train_tfidf_classifier.py --input issues_dataset.jsonl --target category --output model_tfidf_category.joblib
    ```

Transformer (BERT/RoBERTa) Fine-Tuning

    ```bash
    python train_transformer_classifier.py --input issues_dataset.jsonl --target severity --model_name roberta-base --output_dir ./models/transformer_severity
    python train_transformer_classifier.py --input issues_dataset.jsonl --target category --model_name bert-base-uncased --output_dir ./models/transformer_category
    ```

---

## Named Entity Recognition (NER)
Convert your annotated JSONL to spaCy format:

    ```bash
    python ner_converter.py --input ner_annotations.jsonl --output train_data.spacy
    ```
Train or update the NER model:

    ```bash
    python ner_pipeline.py --train --input train_data.spacy --output models/qa_ner --iters 30
    ```

Quick extraction:

    ```bash
    python ner_pipeline.py --extract --model models/qa_ner --text "Error ERR_502 in scheduler on Linux"
    ```

---

## Summarization
    ```bash
    python summarization_pipeline.py --input issues_dataset.jsonl --output issues_summarized.jsonl --extractive_sentences 2 --abstractive_model facebook/bart-large-cnn --max_len 50 --min_len 20 --device -1
    ```

---

## Clustering & Trend Analysis

    ```bash
    python clustering_pipeline.py --input issues_dataset.jsonl --output issues_clustered.jsonl --method hdbscan --min_cluster_size 5 --plot clusters.png
    ```

---

## REST API Integration
Start FastAPI server exposing all pipelines:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

### Endpoints
1. POST /classify
2. POST /ner
3. POST /summarize
3. POST /cluster

---

## Testing the API
Run the integration test suite:

    ```bash
    python test_api.py
    ```

---

## Repository Structure
AutoQAlytics/  
├── fetch_issues.py  
├── etl_pipeline.py  
├── dataset_overview.py  
├── train_tfidf_classifier.py  
├── train_transformer_classifier.py  
├── ner_pipeline.py  
├── summarization_pipeline.py  
├── clustering_pipeline.py  
├── main.py  
├── test_api.py  
├── requirements.txt  
├── LICENSE  
└── README.md  
