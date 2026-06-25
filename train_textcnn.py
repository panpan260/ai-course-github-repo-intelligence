from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


INPUT_PATH = Path("data") / "clean_repos.csv"
MODEL_PATH = Path("models") / "textcnn_model.pth"
TOKENIZER_PATH = Path("models") / "tokenizer.joblib"
CURVE_PATH = Path("results") / "textcnn_training_curve.png"
REPORT_PATH = Path("results") / "classification_report_textcnn.txt"
CONFUSION_MATRIX_PATH = Path("results") / "confusion_matrix_textcnn.png"
COMPARISON_PATH = Path("results") / "model_comparison.csv"

RANDOM_STATE = 42
TARGET = "project_category"
TEXT_FEATURE = "combined_text"
MAX_VOCAB_SIZE = 30000
MIN_FREQ = 2
MAX_LEN = 300
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 128
NUM_FILTERS = 96
KERNEL_SIZES = (3, 4, 5)
DROPOUT = 0.5


def ensure_dirs() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    return re.findall(r"[a-zA-Z0-9_+#.-]+", text)


def build_vocab(texts: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = [
        token
        for token, freq in counter.most_common(MAX_VOCAB_SIZE - 2)
        if freq >= MIN_FREQ
    ]
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for token in most_common:
        word2idx[token] = len(word2idx)
    return word2idx


def encode_text(text: str, word2idx: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    token_ids = [word2idx.get(token, word2idx["<UNK>"]) for token in tokenize(text)]
    token_ids = token_ids[:max_len]
    if len(token_ids) < max_len:
        token_ids.extend([word2idx["<PAD>"]] * (max_len - len(token_ids)))
    return token_ids


class RepoTextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, word2idx: dict[str, int], label2idx: dict[str, int]):
        self.encoded_texts = [encode_text(text, word2idx) for text in texts]
        self.encoded_labels = [label2idx[label] for label in labels]

    def __len__(self) -> int:
        return len(self.encoded_labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.encoded_labels[index], dtype=torch.long),
        )


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = EMBED_DIM,
        num_filters: int = NUM_FILTERS,
        kernel_sizes: tuple[int, ...] = KERNEL_SIZES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pooled = torch.max(activated, dim=2).values
            conv_outputs.append(pooled)
        features = torch.cat(conv_outputs, dim=1)
        features = self.dropout(features)
        return self.fc(features)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    return accuracy_score(all_true, all_pred), all_true, all_pred


def plot_training_curve(history: list[dict[str, float]]) -> None:
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_df["epoch"], history_df["train_accuracy"], marker="s", label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["test_accuracy"], marker="^", label="Test Accuracy")
    plt.title("TextCNN Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(0, max(1.0, float(history_df["train_loss"].max()) + 0.1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=300)
    plt.close()


def append_textcnn_comparison(metrics: dict[str, float]) -> None:
    row = pd.DataFrame(
        [
            {
                "task": "project_category",
                "model": "TextCNN",
                "accuracy": metrics["accuracy"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
            }
        ]
    )
    if COMPARISON_PATH.exists():
        old_df = pd.read_csv(COMPARISON_PATH)
        old_df = old_df[old_df["model"] != "TextCNN"]
        new_df = pd.concat([old_df, row], ignore_index=True)
    else:
        new_df = row
    new_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run clean_data.py first.")

    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(INPUT_PATH)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    class_counts = df[TARGET].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[TARGET].isin(valid_classes)].reset_index(drop=True)
    if df[TARGET].nunique() < 2:
        raise ValueError("Need at least two categories with two or more samples for TextCNN training.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    word2idx = build_vocab(train_df[TEXT_FEATURE])
    labels = sorted(df[TARGET].unique())
    label2idx = {label: index for index, label in enumerate(labels)}
    idx2label = {index: label for label, index in label2idx.items()}

    train_dataset = RepoTextDataset(train_df[TEXT_FEATURE], train_df[TARGET], word2idx, label2idx)
    test_dataset = RepoTextDataset(test_df[TEXT_FEATURE], test_df[TARGET], word2idx, label2idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN(vocab_size=len(word2idx), num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("=" * 72)
    print("Train TextCNN for Project Category Classification")
    print(f"Input file: {INPUT_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Device: {device}")
    print("=" * 72)

    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        all_train_true: list[int] = []
        all_train_pred: list[int] = []

        for input_ids, batch_labels in train_loader:
            input_ids = input_ids.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_train_true.extend(batch_labels.detach().cpu().tolist())
            all_train_pred.extend(preds.detach().cpu().tolist())

        train_loss = total_loss / len(train_dataset)
        train_accuracy = accuracy_score(all_train_true, all_train_pred)
        test_accuracy, _, _ = evaluate_model(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"[EPOCH {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

    _, y_true_idx, y_pred_idx = evaluate_model(model, test_loader, device)
    y_true = [idx2label[index] for index in y_true_idx]
    y_pred = [idx2label[index] for index in y_pred_idx]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    REPORT_PATH.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - TextCNN")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    plot_training_curve(history)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(word2idx),
            "num_classes": len(labels),
            "embed_dim": EMBED_DIM,
            "num_filters": NUM_FILTERS,
            "kernel_sizes": KERNEL_SIZES,
            "dropout": DROPOUT,
            "max_len": MAX_LEN,
            "label2idx": label2idx,
            "idx2label": idx2label,
        },
        MODEL_PATH,
    )
    joblib.dump(
        {
            "word2idx": word2idx,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "max_len": MAX_LEN,
            "token_pattern": r"[a-zA-Z0-9_+#.-]+",
        },
        TOKENIZER_PATH,
    )
    append_textcnn_comparison(metrics)

    print("\n" + "=" * 72)
    print("STEP COMPLETED: train_textcnn.py")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {MODEL_PATH}")
    print(f"Output file: {TOKENIZER_PATH}")
    print(f"Output file: {CURVE_PATH}")
    print(f"Output file: {REPORT_PATH}")
    print(f"Output file: {CONFUSION_MATRIX_PATH}")
    print(f"Output file: {COMPARISON_PATH}")
    print(f"Sample count: {len(df)}")
    print(
        "Key metrics: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision_weighted']:.4f}, "
        f"Recall={metrics['recall_weighted']:.4f}, "
        f"F1={metrics['f1_weighted']:.4f}"
    )
    print("Summary: TextCNN training and evaluation completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
