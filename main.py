#!/usr/bin/env python3
"""
main.py

This FastAPI app unifies all pipelines (classification, NER, summarization, and clustering)
behind a RESTful interface ready for integration.

Usage:
pip install fastapi uvicorn joblib spacy transformers sentence-transformers umap-learn hdbscan networkx nltk
python -m nltk.downloader punkt

uvicorn main:app --host localhost --port 8000 --reload

endpoints:
POST http://localhost:8000/classify { "text": "Error ERR_502 in scheduler on Linux" }
POST http://localhost:8000/ner { "text": "Error ERR_502 in scheduler on Linux" }
POST http://localhost:8000/summarize { "text": "Long bug report body ..." }
POST http://localhost:8000/cluster { "texts": ["Issue one text", "Another issue text", ...] }

"""

# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import spacy
import json
import os

# Transformers & clustering imports
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import hdbscan
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Ensure NLTK punkt is available
nltk.download("punkt", quiet=True)
from nltk import sent_tokenize

app = FastAPI(
    title="AutoQAlytics API",
    description="REST endpoints for classification, NER, summarization & clustering",
    version="1.0"
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD ALL MODELS ON STARTUP
# ──────────────────────────────────────────────────────────────────────────────

# 1.1 Classification (TF-IDF + LR) – adjust paths as needed
sev_bundle = joblib.load("model/model_tfidf_severity.joblib")
clf_sev   = sev_bundle["pipeline"]
le_sev    = sev_bundle["label_encoder"]

cat_bundle = joblib.load("model/model_tfidf_category.joblib")
clf_cat    = cat_bundle["pipeline"]
le_cat     = cat_bundle["label_encoder"]

# 1.2 NER (spaCy)
nlp_ner = spacy.load("model/qa_ner")  # path to your trained NER model

# 1.3 Summarization
summarizer = hf_pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
)

# 1.4 Embedding for clustering
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pydantic Request & Response Schemas
# ──────────────────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    severity: str
    category: str

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class NERResponse(BaseModel):
    entities: List[Entity]

class SummarizationResponse(BaseModel):
    summary_extractive: str
    summary_abstractive: str

class ClusterRequest(BaseModel):
    texts: List[str]

class ClusterResponse(BaseModel):
    umap_x: List[float]
    umap_y: List[float]
    cluster_labels: List[int]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def extractive_summarize(text: str, n_sentences: int = 2) -> str:
    sents = sent_tokenize(text)
    if len(sents) <= n_sentences:
        return text
    vect = TfidfVectorizer().fit_transform(sents)
    sim = (vect * vect.T).toarray()
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_idxs = sorted(idx for idx, _ in ranked[:n_sentences])
    return " ".join(sents[i] for i in top_idxs)

def abstractive_summarize(text: str, max_len: int=50, min_len: int=20) -> str:
    if len(text.split()) < min_len * 1.5:
        return text
    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return out[0]["summary_text"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. API Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResponse)
def classify(req: TextRequest):
    txt = req.text or ""
    # Severity
    y_sev = clf_sev.predict([txt])[0]
    sev  = le_sev.inverse_transform([y_sev])[0]
    # Category
    y_cat = clf_cat.predict([txt])[0]
    cat   = le_cat.inverse_transform([y_cat])[0]
    return ClassificationResponse(severity=sev, category=cat)


@app.post("/ner", response_model=NERResponse)
def ner(req: TextRequest):
    doc = nlp_ner(req.text or "")
    ents = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]
    return NERResponse(entities=ents)


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(req: TextRequest):
    clean = req.text or ""
    ext = extractive_summarize(clean, n_sentences=2)
    abs_ = abstractive_summarize(clean)
    return SummarizationResponse(
        summary_extractive=ext,
        summary_abstractive=abs_
    )


@app.post("/cluster", response_model=ClusterResponse)
def cluster(req: ClusterRequest):
    texts = req.texts or []
    if not texts:
        raise HTTPException(status_code=400, detail="`texts` list cannot be empty")
    # 1) embeddings
    embs = embedder.encode(texts, show_progress_bar=False)
    # 2) reduce dims
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    coords = reducer.fit_transform(embs)
    # 3) HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(embs).tolist()
    return ClusterResponse(
        umap_x=coords[:,0].tolist(),
        umap_y=coords[:,1].tolist(),
        cluster_labels=labels
    )
