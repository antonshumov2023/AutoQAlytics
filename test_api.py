#!/usr/bin/env python3
"""
test_api.py

A small test suite that sends sample requests to the AutoQAlytics FastAPI server
and prints the JSON responses for each endpoint.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_classify():
    url = f"{BASE_URL}/classify"
    payload = {"text": "Error ERR_502 in module scheduler on Linux caused a system crash"}
    resp = requests.post(url, json=payload)
    print("=== /classify ===")
    print("Request:", json.dumps(payload))
    print("Response:", resp.status_code, resp.json(), "\n")

def test_ner():
    url = f"{BASE_URL}/ner"
    payload = {"text": "Error ERR_502 in module scheduler on Linux caused a system crash"}
    resp = requests.post(url, json=payload)
    print("=== /ner ===")
    print("Request:", json.dumps(payload))
    print("Response:", resp.status_code, json.dumps(resp.json(), indent=2), "\n")

def test_summarize():
    url = f"{BASE_URL}/summarize"
    long_text = (
        "When attempting to schedule the training job, DeepSpeed threw Error ERR_502. "
        "The scheduler failed to allocate GPU resources on Linux machine cluster-01, "
        "and the entire training pipeline terminated unexpectedly after 2 hours. "
        "Steps to reproduce: 1) launch job with zero optimization steps, 2) observe ERR_502."
    )
    payload = {"text": long_text}
    resp = requests.post(url, json=payload)
    print("=== /summarize ===")
    print("Request:", long_text, "\n")
    print("Response:", resp.status_code, json.dumps(resp.json(), indent=2), "\n")

def test_cluster():
    url = f"{BASE_URL}/cluster"
    texts = [
        "Error ERR_502 in module scheduler on Linux",
        "Memory leak observed in DeepSpeed optimizer on Ubuntu",
        "Documentation typo in README under installation section",
        "Critical security vulnerability ERR_9001 found in GPU driver",
        "Performance regression: training throughput dropped by 15%"
    ]
    payload = {"texts": texts}
    resp = requests.post(url, json=payload)
    print("=== /cluster ===")
    print("Request texts:")
    for t in texts:
        print("  -", t)
    print("Response:", resp.status_code)
    data = resp.json()
    # print UMAP coords and cluster labels
    for i, (x,y,label) in enumerate(zip(data["umap_x"], data["umap_y"], data["cluster_labels"])):
        print(f"  [{i}] label={label}  ({x:.3f}, {y:.3f})")
    print()

if __name__ == "__main__":
    test_classify()
    test_ner()
    test_summarize()
    test_cluster()
