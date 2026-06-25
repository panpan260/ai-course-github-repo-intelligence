from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data") / "clean_repos.csv"
COMPARISON_PATH = Path("results") / "growth_model_comparison.csv"
REPORT_PATH = Path("results") / "growth_classification_report.txt"
CONFUSION_MATRIX_PATH = Path("results") / "growth_confusion_matrix.png"

RANDOM_STATE = 42
TARGET = "growth_popularity_level"
TEXT_FEATURE = "combined_text"
NUMERIC_FEATURES = [
    "repo_age_days",
    "days_since_update",
    "open_issues_count",
    "size",
    "archived",
    "has_issues",
]


def ensure_dirs() -> None:
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
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
        "MLPClassifier": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=300,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run clean_data.py first.")

    df = pd.read_csv(INPUT_PATH)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    class_counts = df[TARGET].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[TARGET].isin(valid_classes)].reset_index(drop=True)
    if df[TARGET].nunique() < 2:
        raise ValueError("Need at least two growth classes with two or more samples for training.")

    feature_columns = [TEXT_FEATURE, *NUMERIC_FEATURES]
    X = df[feature_columns]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 72)
    print("Train Growth Popularity Prediction Models")
    print(f"Input file: {INPUT_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Target label: {TARGET}")
    print("Leakage check: stargazers_count and watchers_count are NOT used as features.")
    print("=" * 72)

    results = []
    best_model = None
    best_name = ""
    best_f1 = -1.0
    best_pred = None

    for model_name, pipeline in build_models().items():
        print(f"\n[TRAIN] {model_name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        results.append(row)
        print(
            f"Accuracy={row['accuracy']:.4f} | "
            f"Precision={row['precision_weighted']:.4f} | "
            f"Recall={row['recall_weighted']:.4f} | F1={row['f1_weighted']:.4f}"
        )
        if row["f1_weighted"] > best_f1:
            best_f1 = row["f1_weighted"]
            best_name = model_name
            best_model = pipeline
            best_pred = y_pred

    comparison_df = pd.DataFrame(results).sort_values(by=["f1_weighted", "accuracy"], ascending=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    labels = ["low", "medium", "high"]
    labels = [label for label in labels if label in set(y_test) or (best_pred is not None and label in set(best_pred))]
    report = classification_report(y_test, best_pred, labels=labels, zero_division=0)
    REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"Growth Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    del best_model
    print("\n" + "=" * 72)
    print("STEP COMPLETED: train_growth_model.py")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {COMPARISON_PATH}")
    print(f"Output file: {REPORT_PATH}")
    print(f"Output file: {CONFUSION_MATRIX_PATH}")
    print(f"Sample count: {len(df)}")
    print(f"Best growth model: {best_name}")
    print(f"Best weighted F1-score: {best_f1:.4f}")
    print("Summary: Growth popularity prediction completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
