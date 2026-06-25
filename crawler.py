from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = "https://api.github.com/search/repositories"
OUTPUT_PATH = Path("data") / "raw_repos.csv"
TARGET_MAX = 2000
PER_PAGE = 100
MAX_PAGES_PER_KEYWORD = 2

QUERY_KEYWORDS = [
    "computer vision",
    "image classification",
    "object detection",
    "semantic segmentation",
    "natural language processing",
    "large language model",
    "transformer",
    "deep learning",
    "pytorch",
    "tensorflow",
    "robotics",
    "ros",
    "slam",
    "embedded systems",
    "stm32",
    "esp32",
    "arduino",
    "fpga",
    "iot",
    "edge computing",
    "signal processing",
]

CSV_COLUMNS = [
    "repo_name",
    "full_name",
    "description",
    "language",
    "topics",
    "stargazers_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "size",
    "created_at",
    "updated_at",
    "pushed_at",
    "archived",
    "has_issues",
    "license",
    "html_url",
    "api_url",
    "query_keyword",
    "crawl_time",
]


def ensure_dirs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AI-GitHub-Project-Intelligence",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("[INFO] GITHUB_TOKEN detected. Authenticated API mode enabled.")
    else:
        print("[INFO] GITHUB_TOKEN not found. Unauthenticated low-speed mode enabled.")
    return headers


def request_json(
    session: requests.Session,
    headers: dict[str, str],
    params: dict[str, Any],
    max_retries: int = 3,
) -> dict[str, Any] | None:
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(API_URL, headers=headers, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"[WARN] Request error attempt {attempt}/{max_retries}: {exc}")
            time.sleep(5 * attempt)
            continue

        if response.status_code == 200:
            return response.json()

        try:
            message = response.json().get("message", "")
        except ValueError:
            message = response.text[:200]

        if response.status_code in {403, 429}:
            retry_after = response.headers.get("Retry-After")
            reset_at = response.headers.get("X-RateLimit-Reset")
            wait_seconds = 60
            if retry_after and retry_after.isdigit():
                wait_seconds = int(retry_after) + 5
            elif reset_at and reset_at.isdigit():
                wait_seconds = max(int(reset_at) - int(time.time()) + 5, 10)
            print(f"[WARN] Rate limit reached: {message}")
            print(f"[WARN] Sleeping {wait_seconds} seconds before retry...")
            time.sleep(wait_seconds)
            continue

        print(f"[WARN] HTTP {response.status_code}: {message}")
        time.sleep(5 * attempt)

    return None


def normalize_license(repo: dict[str, Any]) -> str:
    license_obj = repo.get("license")
    if isinstance(license_obj, dict):
        return license_obj.get("spdx_id") or license_obj.get("key") or license_obj.get("name") or "unknown"
    return "unknown"


def parse_repo(repo: dict[str, Any], keyword: str, crawl_time: str) -> dict[str, Any]:
    topics = repo.get("topics") or []
    topics_text = "|".join(str(topic) for topic in topics) if isinstance(topics, list) else str(topics)
    return {
        "repo_name": repo.get("name", ""),
        "full_name": repo.get("full_name", ""),
        "description": repo.get("description") or "",
        "language": repo.get("language") or "Unknown",
        "topics": topics_text,
        "stargazers_count": repo.get("stargazers_count", 0),
        "forks_count": repo.get("forks_count", 0),
        "watchers_count": repo.get("watchers_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "size": repo.get("size", 0),
        "created_at": repo.get("created_at", ""),
        "updated_at": repo.get("updated_at", ""),
        "pushed_at": repo.get("pushed_at", ""),
        "archived": repo.get("archived", False),
        "has_issues": repo.get("has_issues", False),
        "license": normalize_license(repo),
        "html_url": repo.get("html_url", ""),
        "api_url": repo.get("url", ""),
        "query_keyword": keyword,
        "crawl_time": crawl_time,
    }


def main() -> None:
    ensure_dirs()
    headers = build_headers()
    delay_seconds = 2 if os.getenv("GITHUB_TOKEN") else 7
    crawl_time = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    repos_by_full_name: dict[str, dict[str, Any]] = {}

    print("=" * 72)
    print("GitHub API Repository Crawler")
    print(f"API endpoint: {API_URL}")
    print(f"Target maximum samples: {TARGET_MAX}")
    print(f"Output file: {OUTPUT_PATH}")
    print("=" * 72)

    for keyword_index, keyword in enumerate(QUERY_KEYWORDS, start=1):
        if len(repos_by_full_name) >= TARGET_MAX:
            break
        print(f"\n[KEYWORD {keyword_index}/{len(QUERY_KEYWORDS)}] {keyword}")
        for page in range(1, MAX_PAGES_PER_KEYWORD + 1):
            if len(repos_by_full_name) >= TARGET_MAX:
                break
            params = {
                "q": f"{keyword} stars:>0",
                "sort": "stars",
                "order": "desc",
                "per_page": PER_PAGE,
                "page": page,
            }
            data = request_json(session, headers, params)
            if not data:
                print(f"[WARN] Skip keyword='{keyword}', page={page}.")
                break

            items = data.get("items", [])
            before_count = len(repos_by_full_name)
            for repo in items:
                if len(repos_by_full_name) >= TARGET_MAX:
                    break
                full_name = repo.get("full_name")
                if full_name and full_name not in repos_by_full_name:
                    repos_by_full_name[full_name] = parse_repo(repo, keyword, crawl_time)

            added = len(repos_by_full_name) - before_count
            print(
                f"[PAGE] keyword='{keyword}' page={page} "
                f"received={len(items)} added={added} total={len(repos_by_full_name)}"
            )
            time.sleep(delay_seconds)

    df = pd.DataFrame(list(repos_by_full_name.values()), columns=CSV_COLUMNS)
    df = df.drop_duplicates(subset=["full_name"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("STEP COMPLETED: crawler.py")
    print("Input file: GitHub REST API")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Final unique samples: {len(df)}")
    print("Summary: Repository metadata collection completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
