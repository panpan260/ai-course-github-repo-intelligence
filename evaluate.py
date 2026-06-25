from __future__ import annotations

from pathlib import Path

import pandas as pd


CLEAN_PATH = Path("data") / "clean_repos.csv"
MODEL_COMPARISON_PATH = Path("results") / "model_comparison.csv"
GROWTH_COMPARISON_PATH = Path("results") / "growth_model_comparison.csv"
ML_REPORT_PATH = Path("results") / "classification_report_ml.txt"
TEXTCNN_REPORT_PATH = Path("results") / "classification_report_textcnn.txt"
GROWTH_REPORT_PATH = Path("results") / "growth_classification_report.txt"
SUMMARY_PATH = Path("results") / "evaluation_summary.txt"


def ensure_dirs() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_report_head(path: Path, max_lines: int = 12) -> str:
    if not path.exists():
        return f"{path}: not found"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[:max_lines])


def describe_best(path: Path, label: str) -> str:
    if not path.exists():
        return f"{label}: comparison file not found"
    df = pd.read_csv(path)
    if df.empty or "f1_weighted" not in df.columns:
        return f"{label}: no valid comparison rows"
    best = df.sort_values(by=["f1_weighted", "accuracy"], ascending=False).iloc[0]
    return (
        f"{label}: best_model={best['model']}, "
        f"accuracy={best['accuracy']:.4f}, "
        f"weighted_f1={best['f1_weighted']:.4f}"
    )


def main() -> None:
    ensure_dirs()
    sample_count = 0
    category_count = "N/A"
    growth_count = "N/A"
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
        sample_count = len(df)
        if "project_category" in df.columns:
            category_count = str(df["project_category"].nunique())
        if "growth_popularity_level" in df.columns:
            growth_count = str(df["growth_popularity_level"].nunique())

    ml_best = describe_best(MODEL_COMPARISON_PATH, "Project category classification")
    growth_best = describe_best(GROWTH_COMPARISON_PATH, "Growth popularity prediction")

    summary_lines = [
        "Evaluation Summary",
        "=" * 72,
        f"Input file: {CLEAN_PATH}",
        f"Sample count: {sample_count}",
        f"Project category classes: {category_count}",
        f"Growth popularity classes: {growth_count}",
        "",
        ml_best,
        growth_best,
        "",
        "Traditional ML report preview:",
        read_report_head(ML_REPORT_PATH),
        "",
        "TextCNN report preview:",
        read_report_head(TEXTCNN_REPORT_PATH),
        "",
        "Growth model report preview:",
        read_report_head(GROWTH_REPORT_PATH),
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 72)
    print("Evaluate Experiment Outputs")
    print(f"Input file: {CLEAN_PATH}")
    print(f"Output file: {SUMMARY_PATH}")
    print(f"Sample count: {sample_count}")
    print(ml_best)
    print(growth_best)
    print("\nReport files checked:")
    print(f"- {ML_REPORT_PATH}")
    print(f"- {TEXTCNN_REPORT_PATH}")
    print(f"- {GROWTH_REPORT_PATH}")
    print("\n" + "=" * 72)
    print("STEP COMPLETED: evaluate.py")
    print("Key results: best model summaries written to evaluation_summary.txt")
    print("Summary: Experiment evaluation summary completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
