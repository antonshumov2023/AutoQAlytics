#!/usr/bin/env python3
"""
train_tfidf_classifier.py

Train & evaluate a TF-IDF → LogisticRegression pipeline on GitHub-issue data.

Usage:
    pip install pandas scikit-learn joblib
    python train_tfidf_classifier.py \
        --input issues_dataset.jsonl \
        --target severity \
        --output model_tfidf_severity.joblib
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

def main(args):
    # 1. Load data
    df = pd.read_json(args.input, lines=True)
    X = df['clean_body'].fillna("")
    y = df[args.target].fillna("Unknown")

    # 2. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # 4. Build pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2),
            max_df=0.8,
            min_df=5,
            max_features=10_000)),
        ('clf', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'))
    ])

    # 5. Hyperparameter tuning (optional)
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'tfidf__max_features': [5_000, 10_000, 20_000]
    }
    grid = GridSearchCV(
        pipe, param_grid, cv=5, n_jobs=-1, verbose=1,
        scoring='f1_weighted')
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_pipe = grid.best_estimator_

    # 6. Evaluation
    y_pred = best_pipe.predict(X_test)
    print(classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred)))

    # 7. Save model + label encoder
    joblib.dump({'pipeline': best_pipe, 'label_encoder': le}, args.output)
    print(f"Saved TF-IDF model to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="JSONL dataset path")
    p.add_argument("--target", required=True,
                   choices=['severity','category'],
                   help="Target column for classification")
    p.add_argument("--output", required=True, help="Where to save the model")
    args = p.parse_args()
    main(args)