from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data") / "repos_with_readme.csv"
OUTPUT_PATH = Path("data") / "clean_repos.csv"

CATEGORY_RULES = {
    "computer_vision": [
        "computer vision",
        "image classification",
        "object detection",
        "semantic segmentation",
        "opencv",
        "yolo",
        "segmentation",
        "classification",
    ],
    "natural_language_processing": [
        "natural language processing",
        "large language model",
        "transformer",
        "nlp",
        "llm",
        "bert",
        "gpt",
        "language model",
    ],
    "deep_learning_framework": [
        "deep learning",
        "pytorch",
        "tensorflow",
        "keras",
        "neural network",
        "machine learning",
    ],
    "robotics": ["robotics", "ros", "slam", "robot", "navigation"],
    "embedded_iot": [
        "embedded systems",
        "stm32",
        "esp32",
        "arduino",
        "fpga",
        "iot",
        "edge computing",
        "microcontroller",
    ],
    "data_science": ["data science", "signal processing", "analytics", "statistics", "data mining"],
}

QUERY_CATEGORY_MAP = {
    keyword: category
    for category, keywords in CATEGORY_RULES.items()
    for keyword in keywords
}


def ensure_dirs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_category(row: pd.Series) -> str:
    query_keyword = normalize_text(row.get("query_keyword", "")).lower()
    if query_keyword in QUERY_CATEGORY_MAP:
        return QUERY_CATEGORY_MAP[query_keyword]

    searchable_text = " ".join(
        [
            normalize_text(row.get("topics", "")),
            normalize_text(row.get("repo_name", "")),
            normalize_text(row.get("description", "")),
            normalize_text(row.get("readme_text", ""))[:3000],
        ]
    ).lower()

    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in searchable_text for keyword in keywords):
            return category
    return "data_science"


def make_quantile_labels(values: pd.Series) -> pd.Series:
    q33 = values.quantile(1 / 3)
    q66 = values.quantile(2 / 3)
    print(f"[INFO] Growth score quantiles: 33%={q33:.4f}, 66%={q66:.4f}")
    conditions = [values <= q33, (values > q33) & (values <= q66), values > q66]
    return pd.Series(np.select(conditions, ["low", "medium", "high"], default="medium"), index=values.index)


def main() -> None:
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run fetch_readme.py first.")

    df = pd.read_csv(INPUT_PATH)
    before_count = len(df)

    print("=" * 72)
    print("Clean Repository Text Dataset")
    print(f"Input file: {INPUT_PATH}")
    print(f"Raw samples: {before_count}")
    print("=" * 72)

    df = df.drop_duplicates(subset=["full_name"]).copy()

    text_columns = ["repo_name", "description", "language", "topics", "readme_text", "license"]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(normalize_text)

    df["language"] = df["language"].replace("", "Unknown")
    df["topics"] = df["topics"].str.replace("|", " ", regex=False)

    if "readme_length" not in df.columns:
        df["readme_length"] = df["readme_text"].str.len()
    df["readme_length"] = pd.to_numeric(df["readme_length"], errors="coerce").fillna(0).astype(int)
    df = df[df["readme_length"] >= 100].copy()

    for col in ["stargazers_count", "forks_count", "watchers_count", "open_issues_count", "size"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["archived", "has_issues"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool).astype(int)

    for col in ["created_at", "updated_at", "pushed_at", "crawl_time"]:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    now = pd.Timestamp.now(tz="UTC")
    df["repo_age_days"] = (now - df["created_at"]).dt.days
    df["days_since_update"] = (now - df["updated_at"]).dt.days
    for col in ["repo_age_days", "days_since_update"]:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0
        df[col] = df[col].fillna(median_value).clip(lower=0).astype(int)

    df["combined_text"] = (
        df["repo_name"]
        + " "
        + df["description"]
        + " "
        + df["topics"]
        + " "
        + df["language"]
        + " "
        + df["readme_text"]
    ).str.strip()

    df["project_category"] = df.apply(infer_category, axis=1)
    df["repo_age_years"] = df["repo_age_days"] / 365.0
    df["growth_score"] = np.log1p(df["stargazers_count"]) / np.log1p(df["repo_age_days"] + 1)
    df["growth_popularity_level"] = make_quantile_labels(df["growth_score"])

    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Samples before cleaning: {before_count}")
    print(f"Samples after cleaning: {len(df)}")
    print("\nProject category counts:")
    print(df["project_category"].value_counts())
    print("\nGrowth popularity counts:")
    print(df["growth_popularity_level"].value_counts().reindex(["low", "medium", "high"], fill_value=0))
    print("\n" + "=" * 72)
    print("STEP COMPLETED: clean_data.py")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Sample count: {len(df)}")
    print("Summary: Dataset cleaning, category labeling, and growth labeling completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
