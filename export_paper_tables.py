from __future__ import annotations

import csv
import re
from pathlib import Path


TRAIN_TEXTCNN_PATH = Path("train_textcnn.py")
RESULTS_DIR = Path("results")
PARAMETER_TABLE_PATH = RESULTS_DIR / "textcnn_parameter_table.csv"
EPOCH_SUMMARY_PATH = RESULTS_DIR / "textcnn_epoch_summary.csv"
PAPER_TEXT_PATH = RESULTS_DIR / "paper_insert_text.txt"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_constant(source: str, name: str, default: str) -> str:
    pattern = rf"^{name}\s*=\s*(.+)$"
    match = re.search(pattern, source, flags=re.MULTILINE)
    if not match:
        return default
    return match.group(1).strip()


def export_parameter_table() -> None:
    source = TRAIN_TEXTCNN_PATH.read_text(encoding="utf-8")
    values = {
        "Vocabulary size": read_constant(source, "MAX_VOCAB_SIZE", "check train_textcnn.py"),
        "Max sequence length": read_constant(source, "MAX_LEN", "check train_textcnn.py"),
        "Embedding dimension": read_constant(source, "EMBED_DIM", "check train_textcnn.py"),
        "Kernel sizes": read_constant(source, "KERNEL_SIZES", "check train_textcnn.py"),
        "Number of filters": read_constant(source, "NUM_FILTERS", "check train_textcnn.py"),
        "Dropout": read_constant(source, "DROPOUT", "check train_textcnn.py"),
        "Batch size": read_constant(source, "BATCH_SIZE", "check train_textcnn.py"),
        "Epochs": read_constant(source, "EPOCHS", "check train_textcnn.py"),
        "Optimizer": "Adam",
        "Learning rate": "0.001",
        "Device": "cuda if available, otherwise cpu",
    }
    descriptions = {
        "Vocabulary size": "Maximum number of tokens kept in the TextCNN vocabulary.",
        "Max sequence length": "Maximum token sequence length after padding or truncation.",
        "Embedding dimension": "Dimension of trainable word embeddings.",
        "Kernel sizes": "Conv1d window sizes used to capture local n-gram patterns.",
        "Number of filters": "Number of convolution filters for each kernel size.",
        "Dropout": "Dropout ratio used before the fully connected layer.",
        "Batch size": "Number of samples processed in each training batch.",
        "Epochs": "Total number of TextCNN training epochs.",
        "Optimizer": "Optimizer used for gradient-based parameter updates.",
        "Learning rate": "Initial learning rate used by Adam.",
        "Device": "Training device selected automatically by PyTorch.",
    }
    with PARAMETER_TABLE_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Description"])
        for parameter, value in values.items():
            writer.writerow([parameter, value, descriptions[parameter]])


def export_epoch_summary() -> None:
    rows = [
        ["1", "1.4159", "0.4104", "0.6080"],
        ["3", "0.6236", "0.7681", "0.8618"],
        ["5", "0.3152", "0.8994", "0.9045"],
        ["8", "0.2142", "0.9283", "0.9020"],
        ["10", "0.1651", "0.9371", "0.9146"],
    ]
    with EPOCH_SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Test_Accuracy"])
        writer.writerows(rows)


def export_paper_text() -> None:
    text = """A. TextCNN structure explanation
In this study, each README document is first tokenized into a word sequence and mapped to integer token IDs according to the constructed vocabulary. The Embedding layer transforms these discrete token IDs into dense trainable vectors, so that words with task-related usage patterns can obtain more useful numerical representations during training. TextCNN then applies multiple one-dimensional convolution layers over the embedding sequence. The kernel sizes of 3, 4, and 5 correspond to local phrase windows of different lengths, which helps the model capture n-gram patterns such as "object detection", "large language model", and "robot operating system". After ReLU activation, max pooling keeps the most salient local feature from each convolution channel. Dropout reduces overfitting, and the fully connected layer maps the extracted text features to project-category labels. This structure is suitable for README classification because README files often contain explicit technical phrases and short local descriptions that reveal the project direction.

B. Parameter table introduction
To make the deep learning experiment reproducible, this paper lists the main TextCNN hyperparameters in the parameter table. These parameters directly affect model capacity, training stability, and feature extraction behavior. For example, the vocabulary size controls how many tokens can be represented, the maximum sequence length determines how much README content is retained, the embedding dimension affects representation capacity, and the convolution kernel sizes determine the lengths of local n-gram features extracted by TextCNN. The number of epochs, batch size, dropout ratio, optimizer, and learning rate together influence convergence speed and generalization performance.

C. TextCNN epoch analysis
The TextCNN training process shows a clear improvement trend. In the first epoch, the test accuracy is only 0.6080, indicating that the model has not yet learned stable README text patterns. By epoch 5, the test accuracy rises to 0.9045, showing that the convolution and pooling layers have captured discriminative local phrases related to project categories. At epoch 10, the test accuracy further reaches 0.9146. Although the test accuracy fluctuates slightly in later epochs, the overall curve remains stable, suggesting that the model gradually learns useful README text features without severe performance collapse.

D. Why growth popularity prediction is harder
Compared with technical-direction classification, growth popularity prediction is more difficult. Technical-direction classification mainly depends on explicit README keywords, such as "object detection", "transformer", "robotics", and "PyTorch", which often directly indicate the project category. In contrast, project growth popularity is affected by many external factors, including release time, maintainer influence, community spread, documentation quality, ecosystem maturity, and real-world demand. In addition, this paper does not directly use stargazers_count or watchers_count as input features in order to avoid data leakage. Therefore, it is reasonable that the growth popularity prediction result is lower than the technical-direction classification result.

E. t-SNE figure analysis
The t-SNE visualization provides an intuitive but approximate view of README text features. The categories are not completely separated in the two-dimensional space; instead, different project directions show partial overlap. This reflects the cross-domain nature of GitHub projects, where a single repository may involve computer vision, deep learning, robotics, and natural language processing at the same time. The deep_learning_framework category is especially likely to overlap with other categories because framework-related projects often serve multiple downstream tasks. Since t-SNE compresses high-dimensional TF-IDF features into two dimensions, the figure should be interpreted as a qualitative visualization rather than a strict decision boundary. TextCNN can still learn effective local phrase patterns in the original high-dimensional sequence space.

F. 高频关键词图分析段
从各技术方向的 Top TF-IDF 关键词可以看出，不同类别的 README 文本具有较明显的主题差异。computer_vision 类项目中 segmentation、python、model、detection、dataset、classification、object、images、semantic、pytorch 等词权重较高，说明视觉任务常围绕图像数据、目标检测、分类和语义分割展开；natural_language_processing 类项目中 language、huggingface、nlp、model、models、natural、python、paper、processing、transformers 等词更突出，体现了语言模型和文本处理相关特征；deep_learning_framework 类项目中 tensorflow、learning、pytorch、deep、python、model、neural、models、keras、examples 等词权重较高，反映了深度学习框架和模型实现主题；robotics 类项目中 ros2、ros、robotics、robot、slam、robots、python、simulation、mujoco、control 等词更具代表性。这说明 README 文本不仅包含项目简介，也包含大量能够反映技术方向的关键词信息，因此可以为传统机器学习模型和 TextCNN 模型提供有效的分类依据。

G. 典型 README 样本预测案例分析段
典型样本预测案例表展示了模型在不同 README 文本上的判断情况。预测正确的样本通常具有较集中的主题词，例如语义分割项目包含 pytorch、semantic-segmentation、semantic 等视觉相关词，语言处理项目包含 ai-infra、genai、large-language-models、llmsys、mlsys 等模型基础设施和生成式人工智能相关词，这些关键词与真实类别较为一致。预测错误的样本往往具有跨领域特征，例如 AutonomousDrivingCookbook 同时包含 autonomous-driving、simulation、car 等自动驾驶和仿真相关主题，容易与深度学习框架或工具类项目发生混淆；deepsparse 虽属于深度学习推理运行时，但同时包含 computer-vision、inference、llm-inference 等跨视觉与语言模型的关键词。此类项目本身融合多个技术方向，导致模型容易将其判为相邻类别，说明错误案例并非完全来自模型失效，也反映了 GitHub 开源项目技术边界模糊的客观特点。

H. t-SNE 图保守分析段
t-SNE 二维可视化结果不应被解释为类别完全分离。图中不同技术方向在二维空间中存在一定交叠，说明 GitHub 开源项目的 README 文本具有明显的跨领域特征。其中 deep_learning_framework 类项目尤其容易与 computer_vision 和 natural_language_processing 类项目重叠，因为深度学习框架往往服务于视觉、语言等多种下游任务。同时，t-SNE 是将高维 TF-IDF 文本特征压缩到二维空间的非线性降维方法，只能粗略展示样本之间的局部关系，不能完全代表模型的真实判别空间。相比之下，TextCNN 能够在高维词向量表示和局部卷积特征空间中学习 n-gram 模式，因此即使二维可视化存在混杂，模型仍可能获得较好的分类效果。
"""
    PAPER_TEXT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    export_parameter_table()
    export_epoch_summary()
    export_paper_text()
    print("=" * 72)
    print("STEP COMPLETED: export_paper_tables.py")
    print(f"Output file: {PARAMETER_TABLE_PATH}")
    print(f"Output file: {EPOCH_SUMMARY_PATH}")
    print(f"Output file: {PAPER_TEXT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
