from __future__ import annotations

from pathlib import Path
import math
import re
from collections import Counter

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None


INPUT_PATH = Path("data") / "clean_repos.csv"
OUTPUT_IMAGE_PATH = Path("results") / "category_keywords_tfidf.png"
OUTPUT_CSV_PATH = Path("results") / "category_keywords_tfidf.csv"
TEXT_COLUMN = "combined_text"
CATEGORY_COLUMN = "project_category"
TOP_N = 10
MAX_FEATURES = 8000
PREFERRED_CATEGORIES = [
    "computer_vision",
    "natural_language_processing",
    "deep_learning_framework",
    "robotics",
]
STOPWORDS = {
    "abs",
    "about",
    "after",
    "alt",
    "align",
    "also",
    "arxiv",
    "assets",
    "badge",
    "and",
    "are",
    "based",
    "been",
    "blob",
    "build",
    "buildstatus",
    "can",
    "code",
    "com",
    "data",
    "docs",
    "file",
    "files",
    "for",
    "from",
    "github",
    "has",
    "have",
    "height",
    "heavy_check_mark",
    "href",
    "html",
    "http",
    "https",
    "icon",
    "image",
    "img",
    "install",
    "into",
    "ipynb",
    "job",
    "license",
    "main",
    "master",
    "more",
    "not",
    "notebook",
    "notebooks",
    "org",
    "pdf",
    "png",
    "project",
    "raw",
    "readme",
    "src",
    "status",
    "shields",
    "svg",
    "that",
    "the",
    "this",
    "use",
    "used",
    "using",
    "version",
    "width",
    "with",
    "www",
    "you",
}


def ensure_dirs() -> None:
    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)


def select_categories(df: pd.DataFrame) -> list[str]:
    available = [category for category in PREFERRED_CATEGORIES if category in set(df[CATEGORY_COLUMN])]
    if len(available) >= 4:
        return available[:4]

    category_counts = df[CATEGORY_COLUMN].value_counts()
    for category in category_counts.index:
        if category not in available:
            available.append(category)
        if len(available) >= 4:
            break
    return available


def build_keyword_table(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    return build_keyword_table_fallback(df, categories)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def build_keyword_table_fallback(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    documents = [tokenize(text) for text in df[TEXT_COLUMN]]
    doc_freq = Counter()
    for tokens in documents:
        doc_freq.update(set(tokens))

    valid_terms = {
        term
        for term, freq in doc_freq.items()
        if freq >= 2 and freq <= len(documents) * 0.8
    }
    if len(valid_terms) > MAX_FEATURES:
        valid_terms = {
            term
            for term, _ in Counter(
                token for tokens in documents for token in tokens if token in valid_terms
            ).most_common(MAX_FEATURES)
        }

    idf = {
        term: math.log((1 + len(documents)) / (1 + doc_freq[term])) + 1
        for term in valid_terms
    }
    category_scores = {category: Counter() for category in categories}
    category_counts = {category: 0 for category in categories}

    for row_index, tokens in enumerate(documents):
        category = df.iloc[row_index][CATEGORY_COLUMN]
        if category not in category_scores:
            continue
        counts = Counter(token for token in tokens if token in valid_terms)
        total_terms = sum(counts.values()) or 1
        category_counts[category] += 1
        for term, count in counts.items():
            category_scores[category][term] += (count / total_terms) * idf[term]

    rows = []
    for category in categories:
        sample_count = max(category_counts[category], 1)
        mean_scores = {
            term: score / sample_count
            for term, score in category_scores[category].items()
        }
        for rank, (keyword, score) in enumerate(
            sorted(mean_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_N],
            start=1,
        ):
            rows.append(
                {
                    "project_category": category,
                    "rank": rank,
                    "keyword": keyword,
                    "mean_tfidf": round(float(score), 6),
                }
            )
    return pd.DataFrame(rows)


def save_keyword_figure(keyword_df: pd.DataFrame, categories: list[str]) -> None:
    if plt is None:
        save_keyword_figure_fallback(keyword_df, categories)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["#2563EB", "#059669", "#DC2626", "#7C3AED"]

    for index, category in enumerate(categories[:4]):
        ax = axes[index]
        category_df = keyword_df[keyword_df["project_category"] == category].sort_values(
            "mean_tfidf", ascending=True
        )
        ax.barh(category_df["keyword"], category_df["mean_tfidf"], color=colors[index])
        ax.set_title(f"Top Keywords: {category}")
        ax.set_xlabel("Mean TF-IDF Weight")
        ax.set_ylabel("Keyword")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    for index in range(len(categories), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Top TF-IDF Keywords by Project Category", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_keyword_figure_fallback(keyword_df: pd.DataFrame, categories: list[str]) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise ModuleNotFoundError("Either matplotlib or Pillow is required to save the keyword figure.")

    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype("arial.ttf", 40) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    heading_font = ImageFont.truetype("arial.ttf", 26) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    text_font = ImageFont.truetype("arial.ttf", 20) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    small_font = ImageFont.truetype("arial.ttf", 17) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    colors = ["#2563EB", "#059669", "#DC2626", "#7C3AED"]

    draw.text((width // 2, 35), "Top TF-IDF Keywords by Project Category", fill="#111827", font=title_font, anchor="ma")

    panel_width, panel_height = 820, 480
    panel_positions = [(60, 120), (930, 120), (60, 650), (930, 650)]
    for index, category in enumerate(categories[:4]):
        x0, y0 = panel_positions[index]
        category_df = keyword_df[keyword_df["project_category"] == category].sort_values("rank")
        max_score = max(category_df["mean_tfidf"].tolist() or [1])
        draw.rectangle([x0, y0, x0 + panel_width, y0 + panel_height], outline="#D0D7DE", width=2)
        draw.text((x0 + 20, y0 + 18), f"Top Keywords: {category}", fill="#111827", font=heading_font)
        draw.text((x0 + 20, y0 + panel_height - 34), "Mean TF-IDF Weight", fill="#374151", font=small_font)

        bar_x = x0 + 250
        bar_max_width = 500
        bar_y = y0 + 68
        row_height = 36
        for row_offset, row in enumerate(category_df.itertuples(index=False)):
            y = bar_y + row_offset * row_height
            keyword = str(row.keyword)[:24]
            score = float(row.mean_tfidf)
            bar_width = int((score / max_score) * bar_max_width) if max_score else 0
            draw.text((x0 + 20, y + 6), keyword, fill="#111827", font=text_font)
            draw.rectangle([bar_x, y + 5, bar_x + bar_width, y + 26], fill=colors[index])
            draw.text((bar_x + bar_width + 8, y + 4), f"{score:.4f}", fill="#374151", font=small_font)

    image.save(OUTPUT_IMAGE_PATH, dpi=(300, 300))


def main() -> None:
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run clean_data.py first.")

    df = pd.read_csv(INPUT_PATH)
    required_columns = {TEXT_COLUMN, CATEGORY_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.dropna(subset=[CATEGORY_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    categories = select_categories(df)
    if not categories:
        raise ValueError("No project categories found for keyword visualization.")

    keyword_df = build_keyword_table(df, categories)
    keyword_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    save_keyword_figure(keyword_df, categories)

    print("=" * 72)
    print("STEP COMPLETED: plot_category_keywords.py")
    print(f"Input file: {INPUT_PATH}")
    for category in categories:
        category_keywords = keyword_df[keyword_df["project_category"] == category]["keyword"].tolist()
        print(f"{category} Top {TOP_N} keywords: {', '.join(category_keywords)}")
    print(f"Output file: {OUTPUT_CSV_PATH}")
    print(f"Output file: {OUTPUT_IMAGE_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
