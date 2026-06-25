from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CLEAN_PATH = Path("data") / "clean_repos.csv"
WRONG_CASES_PATH = Path("results") / "wrong_cases.csv"
TEXTCNN_REPORT_PATH = Path("results") / "classification_report_textcnn.txt"
TEST_PREDICTIONS_PATH = Path("results") / "test_predictions.csv"
OUTPUT_CSV_PATH = Path("results") / "prediction_cases_compact.csv"
OUTPUT_IMAGE_PATH = Path("results") / "prediction_cases_compact.png"
PAPER_TABLE_PATH = Path("results") / "paper_table_ready_cases.csv"
TARGET = "project_category"
RANDOM_STATE = 42


STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "using",
    "based",
    "project",
    "github",
    "readme",
    "awesome",
    "implementation",
    "learning",
}


def ensure_dirs() -> None:
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_keywords(row: pd.Series, min_words: int = 3, max_words: int = 5) -> str:
    source = row.get("topics", "")
    if pd.isna(source) or not str(source).strip():
        source = row.get("description", "")
    if pd.isna(source) or not str(source).strip():
        source = row.get("combined_text", "")

    words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", str(source).replace("|", " "))
    cleaned = []
    for word in words:
        word = word.strip("._-").lower()
        if not word or word in STOPWORDS or word in cleaned:
            continue
        cleaned.append(word)
        if len(cleaned) >= max_words:
            break
    if len(cleaned) < min_words:
        fallback = str(row.get("query_keyword", "")).replace("_", " ").split()
        for word in fallback:
            word = word.lower()
            if word not in cleaned:
                cleaned.append(word)
            if len(cleaned) >= min_words:
                break
    return ", ".join(cleaned[:max_words])


def brief_reason(row: pd.Series, result: str) -> str:
    true_label = str(row.get("true_label", row.get(TARGET, "")))
    pred_label = str(row.get("pred_label", true_label))
    if result == "correct":
        return f"Keywords match {true_label}."
    return f"Cross-domain terms cause {pred_label}."


def chinese_analysis(row: pd.Series, result: str) -> str:
    true_label = str(row.get("true_label", row.get(TARGET, "")))
    pred_label = str(row.get("pred_label", row.get(TARGET, "")))
    if result == "correct":
        if true_label == "computer_vision":
            return "视觉特征明显"
        if true_label == "natural_language_processing":
            return "语言模型特征明显"
        if true_label == "deep_learning_framework":
            return "框架特征明显"
        if true_label == "robotics":
            return "机器人特征明显"
        return "类别特征明显"
    if true_label == "robotics" and pred_label == "deep_learning_framework":
        return "跨机器人与深度学习"
    if pred_label == "natural_language_processing":
        return "LLM关键词造成混淆"
    if pred_label == "computer_vision":
        return "视觉关键词造成混淆"
    if pred_label == "deep_learning_framework":
        return "框架关键词造成混淆"
    return "跨领域特征混淆"


def load_wrong_cases(count: int = 2) -> pd.DataFrame:
    if not WRONG_CASES_PATH.exists():
        return pd.DataFrame()
    wrong_df = pd.read_csv(WRONG_CASES_PATH).head(count).copy()
    wrong_df["Result"] = "wrong"
    return wrong_df


def load_correct_cases(clean_df: pd.DataFrame, count: int = 2) -> pd.DataFrame:
    if TEST_PREDICTIONS_PATH.exists():
        test_df = pd.read_csv(TEST_PREDICTIONS_PATH)
        if {"true_label", "pred_label"}.issubset(test_df.columns):
            correct_df = test_df[test_df["true_label"] == test_df["pred_label"]].head(count).copy()
            if len(correct_df) >= count:
                correct_df["Result"] = "correct"
                return correct_df

    correct_parts = []
    for label in ["computer_vision", "natural_language_processing", "deep_learning_framework", "robotics"]:
        label_df = clean_df[clean_df[TARGET] == label]
        if label_df.empty:
            continue
        row = label_df.sample(n=1, random_state=RANDOM_STATE + len(correct_parts)).copy()
        row["true_label"] = label
        row["pred_label"] = label
        row["Result"] = "correct"
        correct_parts.append(row)
        if len(correct_parts) >= count:
            break
    if not correct_parts:
        return pd.DataFrame()
    return pd.concat(correct_parts, ignore_index=True).head(count)


def normalize_cases(cases_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases_df.iterrows():
        result = str(row.get("Result", ""))
        rows.append(
            {
                "Repository": row.get("full_name", ""),
                "Keywords": extract_keywords(row),
                "True_Label": row.get("true_label", row.get(TARGET, "")),
                "Predicted_Label": row.get("pred_label", row.get(TARGET, "")),
                "Analysis": brief_reason(row, result),
            }
        )
    return pd.DataFrame(rows)


def normalize_cases_for_word(cases_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases_df.iterrows():
        result = str(row.get("Result", ""))
        rows.append(
            {
                "仓库名称": row.get("full_name", ""),
                "关键词/主题": extract_keywords(row),
                "真实类别": row.get("true_label", row.get(TARGET, "")),
                "预测类别": row.get("pred_label", row.get(TARGET, "")),
                "分析": chinese_analysis(row, result),
            }
        )
    return pd.DataFrame(rows)


def save_compact_image(display_df: pd.DataFrame) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed; skipped prediction_cases_compact.png.")
        return False

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.axis("off")
    ax.set_title("Typical README Prediction Cases", fontsize=12, fontweight="bold", pad=8)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.22, 0.18, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.9)
    table.scale(1.0, 1.55)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row_index == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold", ha="center")
            continue

        is_correct = display_df.iloc[row_index - 1]["True_Label"] == display_df.iloc[row_index - 1]["Predicted_Label"]
        cell.set_facecolor("#E8F5E9" if is_correct else "#FFF3E0")
        if col_index in {2, 3}:
            cell.set_text_props(ha="center")
        else:
            cell.set_text_props(ha="left")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    ensure_dirs()
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CLEAN_PATH}. Run clean_data.py first.")

    clean_df = pd.read_csv(CLEAN_PATH)
    wrong_cases = load_wrong_cases(count=2)
    correct_cases = load_correct_cases(clean_df, count=2)

    if len(correct_cases) < 2 or len(wrong_cases) < 2:
        raise ValueError("Need at least two correct and two wrong display cases.")

    if not TEST_PREDICTIONS_PATH.exists():
        print("[INFO] Using available wrong_cases and clean samples for qualitative case display.")
    elif not TEXTCNN_REPORT_PATH.exists():
        print("[INFO] TextCNN report not found; using available case files for qualitative display.")

    final_df = pd.concat(
        [normalize_cases(correct_cases), normalize_cases(wrong_cases)],
        ignore_index=True,
    )
    word_df = pd.concat(
        [normalize_cases_for_word(correct_cases), normalize_cases_for_word(wrong_cases)],
        ignore_index=True,
    )
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    word_df.to_csv(PAPER_TABLE_PATH, index=False, encoding="utf-8-sig")
    image_saved = save_compact_image(final_df)

    print("=" * 72)
    print("STEP COMPLETED: export_prediction_cases_compact.py")
    print(f"Output file: {OUTPUT_CSV_PATH}")
    print(f"Output file: {PAPER_TABLE_PATH}")
    if image_saved:
        print(f"Output file: {OUTPUT_IMAGE_PATH}")
    print("For Word paper, it is recommended to copy prediction_cases_compact.csv into a Word table instead of inserting the image.")
    print("=" * 72)


if __name__ == "__main__":
    main()
