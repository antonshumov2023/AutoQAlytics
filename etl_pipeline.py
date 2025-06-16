#!/usr/bin/env python3
"""
etl_pipeline.py

ETL pipeline for GitHub issue JSON → cleaned, deduped, preprocessed dataset
with added classification fields (severity, category, resolution time).

Usage:
    pip install pandas spacy beautifulsoup4
    python -m spacy download en_core_web_sm
    python etl_pipeline.py \
        --input google_guava_issues.json \
        --output issues_dataset.jsonl
"""

import json
import re
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import spacy

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load spaCy model (disable unused components for speed)
# ──────────────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cleaning & Preprocessing Helpers
# ──────────────────────────────────────────────────────────────────────────────
def clean_markdown(text: str) -> str:
    """Remove code fences, HTML tags, excessive whitespace."""
    text = re.sub(r"```[\s\S]+?```", "", text)      # strip ``` code fences
    text = re.sub(r"`([^`\n]+)`", r"\1", text)       # inline code
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def tokenize_and_lemmatize(text: str):
    """Return tokens & lemmas, excluding stopwords/punctuation."""
    doc = nlp(text.lower())
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct and t.text.strip()]
    lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.lemma_.strip()]
    return tokens, lemmas


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classification Mappings
# ──────────────────────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    "blocker":    "Blocker",
    "critical":   "Critical",
    "major":      "Major",
    "minor":      "Minor",
    "trivial":    "Trivial",
    "p0":         "Blocker",
    "p1":         "Critical",
    "p2":         "Major",
    "p3":         "Minor",
    "p4":         "Trivial",
}

CATEGORY_MAP = {
    "bug":                  "Bug",
    "enhancement":          "Enhancement",
    "feature":              "Feature",
    "documentation":        "Documentation",
    "perf":                 "Performance",
    "performance":          "Performance",
    "security":             "Security",
    "type=defect":          "Bug",
    "type=enhancement":     "Enhancement",
    "type=addition":        "Feature",
    "type=debeta":          "Feature",
    "type=api-docs":        "Documentation",
    "type=documentation":   "Documentation",
    "type=performance":     "Performance",
    "type=other":           "Other",
}

def map_severity(labels):
    for name in labels:
        key = name.lower()
        if key in SEVERITY_MAP:
            return SEVERITY_MAP[key]
    return None #"Unknown"


def map_category(labels):
    for name in labels:
        key = name.lower()
        if key in CATEGORY_MAP:
            return CATEGORY_MAP[key]
    return None #"Other"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main ETL Logic
# ──────────────────────────────────────────────────────────────────────────────
def etl(input_path: str, output_path: str):
    # 4.1 Load raw JSON
    with open(input_path, "r", encoding="utf-8") as f:
        raw_issues = json.load(f)

    processed = []
    seen_ids = set()

    for issue in raw_issues:
        # skip pull requests
        if issue.get("pull_request"):
            continue

        issue_id = issue["id"]
        if issue_id in seen_ids:
            continue  # dedupe by GitHub issue ID
        seen_ids.add(issue_id)

        # extract core fields
        number     = issue["number"]
        created_at = issue["created_at"]
        closed_at  = issue.get("closed_at")
        title      = issue.get("title", "").strip()
        body       = issue.get("body") or ""
        labels     = [lbl["name"] for lbl in issue.get("labels", [])]

        # clean & preprocess text
        clean_body = clean_markdown(body)
        tokens, lemmas = tokenize_and_lemmatize(clean_body)

        # classification fields
        severity = map_severity(labels)
        category = map_category(labels)
        #skip issues without severity / category
        if severity is None or category is None:
            continue

        # compute resolution time (days)
        resolution_days = None
        if closed_at:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            try:
                d1 = datetime.strptime(created_at, fmt)
                d2 = datetime.strptime(closed_at, fmt)
                resolution_days = (d2 - d1).days
            except ValueError:
                pass

        # assemble record
        processed.append({
            "issue_id":        issue_id,
            "number":          number,
            "title":           title,
            "raw_body":        body,
            "clean_body":      clean_body,
            "tokens":          tokens,
            "lemmas":          lemmas,
            "labels":          labels,
            "severity":        severity,
            "category":        category,
            "created_at":      created_at,
            "closed_at":       closed_at,
            "resolution_days": resolution_days
        })

    # 4.2 Convert to DataFrame (optional)
    df = pd.DataFrame(processed)

    # 4.3 Persist to disk (JSON Lines for easy downstream loading)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"ETL complete! {len(df)} issues written to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for GitHub issues")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to raw issues JSON (from fetch_issues.py)")
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write cleaned JSONL dataset")
    args = parser.parse_args()

    etl(args.input, args.output)
