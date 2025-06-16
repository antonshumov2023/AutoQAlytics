#!/usr/bin/env python3
"""
clustering_pipeline.py

Compute semantic embeddings for issue texts, reduce dimensions for viz,
cluster via KMeans and HDBSCAN, evaluate cluster quality, and save results.

Usage:
    pip install pandas sentence-transformers umap-learn hdbscan scikit-learn matplotlib
    python clustering_pipeline.py \
      --input issues_dataset.jsonl \
      --output issues_clustered.jsonl \
      --method hdbscan \
      --min_cluster_size 5
"""

import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import matplotlib.pyplot as plt

def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Compute SBERT embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

def reduce_dims(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """Project embeddings into 2D with UMAP for visualization."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)

def cluster_kmeans(embeddings, n_clusters=10, random_state=42):
    """Cluster embeddings with KMeans and return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    return labels, score

def cluster_hdbscan(embeddings, min_cluster_size=5, metric="euclidean"):
    """Cluster embeddings with HDBSCAN and return labels."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)
    # silhouette_score requires >1 cluster; skip if only noise or 1 cluster
    score = None
    if len(set(labels)) > 1 and any(l >= 0 for l in labels):
        # remove noise (-1) for silhouette
        mask = labels >= 0
        score = silhouette_score(embeddings[mask], labels[mask])
    return labels, score

def plot_clusters(umap_coords, labels, title="Clusters", output_path=None):
    """Scatter-plot 2D UMAP coords colored by cluster label."""
    plt.figure(figsize=(8,6))
    sc = plt.scatter(
        umap_coords[:,0], umap_coords[:,1],
        c=labels, cmap="tab20", s=10, alpha=0.8
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(sc, label="cluster")
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close()

def main(args):
    # 1) Load data
    df = pd.read_json(args.input, lines=True)
    texts = df["clean_body"].fillna("").tolist()

    # 2) Embedding
    print("Computing embeddings…")
    embeddings = embed_texts(texts, model_name=args.embed_model)

    # 3) Dimensionality reduction for visualization
    print("Reducing dimensions with UMAP…")
    umap_coords = reduce_dims(
        embeddings,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist
    )
    df["umap_x"], df["umap_y"] = umap_coords[:,0], umap_coords[:,1]

    # 4) Clustering
    if args.method == "kmeans":
        print(f"Clustering with KMeans (k={args.k})…")
        labels, score = cluster_kmeans(embeddings, n_clusters=args.k)
        df["cluster_kmeans"] = labels
        print(f"KMeans silhouette score: {score:.4f}")
        plot_clusters(umap_coords, labels, title="KMeans Clusters", output_path=args.plot)
    else:
        print(f"Clustering with HDBSCAN (min_size={args.min_cluster_size})…")
        labels, score = cluster_hdbscan(
            embeddings,
            min_cluster_size=args.min_cluster_size,
            metric=args.hdbscan_metric
        )
        df["cluster_hdbscan"] = labels
        print(f"HDBSCAN silhouette score: {score}")
        plot_clusters(umap_coords, labels, title="HDBSCAN Clusters", output_path=args.plot)

    # 5) Save output
    print(f"Saving clustered dataset to {args.output}")
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clustering pipeline for issue texts")
    p.add_argument("--input",            "-i", required=True, help="JSONL input file")
    p.add_argument("--output",           "-o", required=True, help="JSONL output file")
    p.add_argument("--method",           "-m",
        choices=["kmeans","hdbscan"], default="hdbscan",
        help="Clustering algorithm")
    p.add_argument("--k", type=int,      default=10,              help="# clusters for KMeans")
    p.add_argument("--min_cluster_size", type=int, default=5,      help="HDBSCAN min cluster size")
    p.add_argument("--hdbscan_metric",   default="euclidean",      help="Metric for HDBSCAN")
    p.add_argument("--embed_model",      default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    p.add_argument("--umap_neighbors",   type=int, default=15,      help="UMAP n_neighbors")
    p.add_argument("--umap_min_dist",    type=float, default=0.1,   help="UMAP min_dist")
    p.add_argument("--plot",             help="Optional path to save UMAP scatter plot")
    args = p.parse_args()
    main(args)
