from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


INPUT_PATH = Path("data") / "raw_repos.csv"
OUTPUT_PATH = Path("data") / "repos_with_readme.csv"
README_API_TEMPLATE = "https://api.github.com/repos/{full_name}/readme"


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


def request_readme(
    session: requests.Session,
    full_name: str,
    headers: dict[str, str],
    max_retries: int = 3,
) -> tuple[str, bool]:
    url = README_API_TEMPLATE.format(full_name=full_name)
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            print(f"[WARN] README request error for {full_name}: {exc}")
            time.sleep(5 * attempt)
            continue

        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            content = data.get("content", "")
            encoding = data.get("encoding", "")
            if encoding == "base64" and content:
                try:
                    text = base64.b64decode(content).decode("utf-8", errors="ignore")
                    return text, True
                except Exception as exc:
                    print(f"[WARN] Base64 decode failed for {full_name}: {exc}")
                    return "", False
            return "", False

        if response.status_code == 404:
            return "", False

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
            print(f"[WARN] Rate limit while fetching README: {message}")
            print(f"[WARN] Sleeping {wait_seconds} seconds before retry...")
            time.sleep(wait_seconds)
            continue

        print(f"[WARN] HTTP {response.status_code} for {full_name}: {message}")
        time.sleep(5 * attempt)

    return "", False


def main() -> None:
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run crawler.py first.")

    df = pd.read_csv(INPUT_PATH)
    headers = build_headers()
    delay_seconds = 1.5 if os.getenv("GITHUB_TOKEN") else 6
    session = requests.Session()

    readme_texts: list[str] = []
    success_count = 0
    failed_count = 0

    print("=" * 72)
    print("Fetch GitHub Repository README Files")
    print(f"Input file: {INPUT_PATH}")
    print(f"Repositories: {len(df)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("=" * 72)

    for index, full_name in enumerate(df["full_name"].fillna(""), start=1):
        if not full_name:
            readme_texts.append("")
            failed_count += 1
            continue

        text, ok = request_readme(session, str(full_name), headers)
        readme_texts.append(text)
        if ok and text.strip():
            success_count += 1
        else:
            failed_count += 1

        if index % 20 == 0 or index == len(df):
            print(
                f"[README] processed={index}/{len(df)} "
                f"success={success_count} failed={failed_count}"
            )
        time.sleep(delay_seconds)

    df["readme_text"] = readme_texts
    df["readme_length"] = df["readme_text"].fillna("").str.len()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("STEP COMPLETED: fetch_readme.py")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Sample count: {len(df)}")
    print(f"README success count: {success_count}")
    print(f"README failed or empty count: {failed_count}")
    print("Summary: README text collection completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
