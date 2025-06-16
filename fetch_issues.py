#!/usr/bin/env python3
"""
fetch_issues.py

Fetches all issues (open & closed) from a given GitHub repo and saves to JSON.

Usage:
    export GITHUB_TOKEN="ghp_xxx…"       # your personal access token
    python fetch_issues.py
"""

import os
import sys
import time
import json
import requests

# 1) CONFIGURATION
OWNER   = "google"
REPO    = "guava"
OUTFILE = f"dataset\{OWNER}_{REPO}_issues.json"
TOKEN   = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    print("ERROR: set env var GITHUB_TOKEN with repo:issues scope")
    sys.exit(1)

# 2) HELPER TO FETCH A PAGE OF ISSUES
def fetch_issues_page(page:int=1, per_page:int=100):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
    headers = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "state": "all",      # fetch open + closed
        "per_page": per_page,
        "page": page
    }
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error [{resp.status_code}]: {resp.text}")
    return resp.json(), resp.headers

# 3) PAGINATION LOOP
def fetch_all_issues():
    all_issues = []
    page = 1

    while True:
        issues, headers = fetch_issues_page(page=page)
        if not issues:
            break
        all_issues.extend(issues)
        print(f"  • Fetched page {page}, {len(issues)} issues (total: {len(all_issues)})")
        
        # rate-limit handling
        remaining = int(headers.get("X-RateLimit-Remaining", 0))
        reset_time = int(headers.get("X-RateLimit-Reset", time.time()))
        if remaining < 10:
            wait_sec = max(reset_time - time.time(), 0) + 2
            print(f"Rate limit nearly exhausted. Sleeping for {int(wait_sec)}s…")
            time.sleep(wait_sec)

        page += 1

    return all_issues

# 4) RUN & DUMP
if __name__ == "__main__":
    print(f"Starting fetch of all issues from {OWNER}/{REPO}…")
    issues = fetch_all_issues()
    print(f"Total issues fetched: {len(issues)}")

    # Save to file
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)
    print(f"Issues written to {OUTFILE}")