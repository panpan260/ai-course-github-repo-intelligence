from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CLEAN_PATH = Path("data") / "clean_repos.csv"
RESULTS_DIR = Path("results")
CATEGORY_COUNT_PATH = RESULTS_DIR / "category_count.png"
LANGUAGE_DISTRIBUTION_PATH = RESULTS_DIR / "language_distribution.png"
README_LENGTH_DISTRIBUTION_PATH = RESULTS_DIR / "readme_length_distribution.png"
GROWTH_SCORE_DISTRIBUTION_PATH = RESULTS_DIR / "growth_score_distribution.png"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_PNG = RESULTS_DIR / "model_comparison.png"
TEXT_FEATURE_TSNE_PATH = RESULTS_DIR / "text_feature_tsne_balanced.png"
PREDICTION_CASES_COMPACT_PATH = RESULTS_DIR / "prediction_cases_compact.png"
CATEGORY_KEYWORDS_PATH = RESULTS_DIR / "category_keywords_tfidf.png"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_category_count(df: pd.DataFrame) -> None:
    counts = df["project_category"].value_counts()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#2563EB")
    plt.title("Project Category Distribution")
    plt.xlabel("Repository Count")
    plt.ylabel("Project Category")
    plt.tight_layout()
    plt.savefig(CATEGORY_COUNT_PATH, dpi=300)
    plt.close()


def save_language_distribution(df: pd.DataFrame) -> None:
    counts = df["language"].fillna("Unknown").replace("", "Unknown").value_counts().head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#16A34A")
    plt.title("Top 10 Programming Languages")
    plt.xlabel("Repository Count")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(LANGUAGE_DISTRIBUTION_PATH, dpi=300)
    plt.close()


def save_readme_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df["readme_length"]), bins=30, kde=True, color="#EA580C")
    plt.title("README Length Distribution")
    plt.xlabel("log1p(README Length)")
    plt.ylabel("Repository Count")
    plt.tight_layout()
    plt.savefig(README_LENGTH_DISTRIBUTION_PATH, dpi=300)
    plt.close()


def save_growth_score_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["growth_score"], bins=30, kde=True, color="#7C3AED")
    plt.title("Growth Score Distribution")
    plt.xlabel("Growth Score")
    plt.ylabel("Repository Count")
    plt.tight_layout()
    plt.savefig(GROWTH_SCORE_DISTRIBUTION_PATH, dpi=300)
    plt.close()


def save_model_comparison() -> bool:
    if not MODEL_COMPARISON_CSV.exists():
        return False
    comparison_df = pd.read_csv(MODEL_COMPARISON_CSV)
    if comparison_df.empty or "model" not in comparison_df.columns or "f1_weighted" not in comparison_df.columns:
        return False
    comparison_df = comparison_df.sort_values(by="f1_weighted", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison_df, x="model", y="f1_weighted", color="#0F766E")
    plt.title("Traditional ML and TextCNN Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Weighted F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PNG, dpi=300)
    plt.close()
    return True


def main() -> None:
    ensure_dirs()
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CLEAN_PATH}. Run clean_data.py first.")

    df = pd.read_csv(CLEAN_PATH)
    sns.set_theme(style="whitegrid")

    save_category_count(df)
    save_language_distribution(df)
    save_readme_length_distribution(df)
    save_growth_score_distribution(df)
    comparison_saved = save_model_comparison()

    print("=" * 72)
    print("Generate Visualization Figures")
    print(f"Input file: {CLEAN_PATH}")
    print(f"Output file: {CATEGORY_COUNT_PATH}")
    print(f"Output file: {LANGUAGE_DISTRIBUTION_PATH}")
    print(f"Output file: {README_LENGTH_DISTRIBUTION_PATH}")
    print(f"Output file: {GROWTH_SCORE_DISTRIBUTION_PATH}")
    if comparison_saved:
        print(f"Output file: {MODEL_COMPARISON_PNG}")
    else:
        print(f"[WARN] Model comparison file not found or invalid: {MODEL_COMPARISON_CSV}")
    if TEXT_FEATURE_TSNE_PATH.exists():
        print(f"Additional figure found: {TEXT_FEATURE_TSNE_PATH}")
    if PREDICTION_CASES_COMPACT_PATH.exists():
        print(f"Additional figure found: {PREDICTION_CASES_COMPACT_PATH}")
    if CATEGORY_KEYWORDS_PATH.exists():
        print(f"Additional figure found: {CATEGORY_KEYWORDS_PATH}")
    print(f"Sample count: {len(df)}")
    print(f"Project category count: {df['project_category'].nunique()}")
    print(f"Language count: {df['language'].nunique()}")
    print("\n" + "=" * 72)
    print("STEP COMPLETED: visualize.py")
    print("Key results: paper-ready figures have been generated in results/.")
    print("Summary: Dataset and model visualization completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
