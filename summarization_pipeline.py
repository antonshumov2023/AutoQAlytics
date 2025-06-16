#!/usr/bin/env python3
"""
summarization_pipeline.py

Reads issues_dataset.jsonl, summarizes each issue’s clean_body
extractively via TextRank and abstractively via a pretrained BART model,
and outputs issues_summarized.jsonl with summaries added.

Usage:
    pip install pandas scikit-learn networkx nltk transformers torch tqdm
    python -m nltk.downloader punkt
    python summarization_pipeline.py \
      --input issues_dataset.jsonl \
      --output issues_summarized.jsonl \
      --extractive_sentences 2 \
      --abstractive_model facebook/bart-large-cnn \
      --max_len 50 \
      --min_len 20
"""

import json
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────────────
# 1) EXTRACTIVE: TextRank over sentence similarities
# ──────────────────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
sent_tokenize = nltk.sent_tokenize

def extractive_summarize(text, n_sentences=2):
    sents = sent_tokenize(text)
    if len(sents) <= n_sentences:
        return text  # nothing to reduce

    # Vectorize sentences
    vect = TfidfVectorizer().fit_transform(sents)
    sim_matrix = (vect * vect.T).toarray()

    # Build graph & rank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank & select top n_sentences
    ranked = sorted(((scores[i], i) for i in scores), reverse=True)
    selected_idxs = sorted(idx for _, idx in ranked[:n_sentences])
    return " ".join(sents[i] for i in selected_idxs)

# ──────────────────────────────────────────────────────────────────────────────
# 2) ABSTRACTIVE: HuggingFace summarization pipeline
# ──────────────────────────────────────────────────────────────────────────────
def make_abstractive_summarizer(model_name, device=-1):
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=device,
        clean_up_tokenization_spaces=True
    )

def abstractive_summarize(summarizer, text, max_len=50, min_len=20):
    # some texts are short—avoid over-summarizing
    if len(text.split()) < min_len * 1.5:
        return text
    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return out[0]['summary_text']

# ──────────────────────────────────────────────────────────────────────────────
# 3) MAIN ETL
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    # Load dataset
    df = pd.read_json(args.input, lines=True)

    # Prepare abstractive model once
    abstractive = make_abstractive_summarizer(
        args.abstractive_model,
        device=args.device
    )

    # Apply summarization
    summaries = []
    for text in tqdm(df['clean_body'].fillna(""), desc="Summarizing"):
        ext = extractive_summarize(text, args.extractive_sentences)
        abs_ = abstractive_summarize(abstractive, text, args.max_len, args.min_len)
        summaries.append((ext, abs_))

    # Unpack into new columns
    df['summary_extractive'], df['summary_abstractive'] = zip(*summaries)

    # Write out
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"\nWrote summarized dataset to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Issue summarization pipeline")
    p.add_argument("--input",  "-i", required=True, help="Input JSONL (issues_dataset.jsonl)")
    p.add_argument("--output", "-o", required=True, help="Output JSONL with summaries")
    p.add_argument("--extractive_sentences", type=int, default=2,
                   help="How many sentences for extractive summary")
    p.add_argument("--abstractive_model", default="facebook/bart-large-cnn",
                   help="HuggingFace model for abstractive summarization")
    p.add_argument("--max_len", type=int, default=50, help="Max tokens (abstractive)")
    p.add_argument("--min_len", type=int, default=20, help="Min tokens (abstractive)")
    p.add_argument("--device", type=int, default=-1,
                   help="Device for transformer (−1=CPU, ≥0=CUDA device ID)")
    args = p.parse_args()
    main(args)
