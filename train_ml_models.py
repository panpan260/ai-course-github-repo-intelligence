from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


INPUT_PATH = Path("data") / "clean_repos.csv"
MODEL_PATH = Path("models") / "best_ml_model.joblib"
COMPARISON_PATH = Path("results") / "model_comparison.csv"
COMPARISON_FIG_PATH = Path("results") / "model_comparison.png"
REPORT_PATH = Path("results") / "classification_report_ml.txt"
CONFUSION_MATRIX_PATH = Path("results") / "confusion_matrix_ml.png"
WRONG_CASES_PATH = Path("results") / "wrong_cases.csv"

RANDOM_STATE = 42
TARGET = "project_category"
TEXT_FEATURE = "combined_text"


def ensure_dirs() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run clean_data.py first.")
    df = pd.read_csv(INPUT_PATH)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    class_counts = df[TARGET].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[TARGET].isin(valid_classes)].reset_index(drop=True)
    if df[TARGET].nunique() < 2:
        raise ValueError("Need at least two categories with two or more samples for training.")
    return df


def build_models() -> dict[str, Pipeline]:
    return {
        "KNN": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
                ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison_df, x="model", y="f1_weighted", color="#3B82F6")
    plt.title("Traditional ML Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Weighted F1-score")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(COMPARISON_FIG_PATH, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    X = df[TEXT_FEATURE]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 72)
    print("Train Traditional ML Models for Project Category Classification")
    print(f"Input file: {INPUT_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Target label: {TARGET}")
    print(f"Input feature: {TEXT_FEATURE}")
    print("=" * 72)

    results = []
    trained_models = {}
    for model_name, model in build_models().items():
        print(f"\n[TRAIN] {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        row = {
            "task": "project_category",
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        results.append(row)
        trained_models[model_name] = model
        print(
            f"Accuracy={row['accuracy']:.4f} | "
            f"Precision={row['precision_weighted']:.4f} | "
            f"Recall={row['recall_weighted']:.4f} | F1={row['f1_weighted']:.4f}"
        )

    comparison_df = pd.DataFrame(results).sort_values(by=["f1_weighted", "accuracy"], ascending=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    plot_model_comparison(comparison_df)

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_model,
            "target": TARGET,
            "feature": TEXT_FEATURE,
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
        },
        MODEL_PATH,
    )

    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred_best, labels=labels, zero_division=0)
    REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred_best, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    test_df = df.loc[X_test.index].copy()
    test_df["true_label"] = y_test.values
    test_df["pred_label"] = y_pred_best
    wrong_df = test_df[test_df["true_label"] != test_df["pred_label"]].copy()
    wrong_columns = [
        "full_name",
        "description",
        "language",
        "topics",
        "query_keyword",
        "project_category",
        "true_label",
        "pred_label",
        "html_url",
    ]
    wrong_df[wrong_columns].to_csv(WRONG_CASES_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("STEP COMPLETED: train_ml_models.py")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {COMPARISON_PATH}")
    print(f"Output file: {COMPARISON_FIG_PATH}")
    print(f"Output file: {MODEL_PATH}")
    print(f"Output file: {REPORT_PATH}")
    print(f"Output file: {CONFUSION_MATRIX_PATH}")
    print(f"Output file: {WRONG_CASES_PATH}")
    print(f"Sample count: {len(df)}")
    print(f"Best traditional ML model: {best_model_name}")
    print(f"Best weighted F1-score: {comparison_df.iloc[0]['f1_weighted']:.4f}")
    print("Summary: Traditional ML comparison completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
