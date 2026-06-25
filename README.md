# 基于 GitHub API 与 TextCNN 的开源项目技术方向识别与成长热度预测研究

本项目是在 GitHub API 开源项目热度预测项目基础上的升级版。项目保留真实 GitHub REST API 数据采集主线，并进一步采集仓库 README 文本，构建自定义开源项目文本数据集，用于人工智能课程论文实验展示。

项目完成两个任务：

1. 开源项目技术方向分类：预测仓库属于 `computer_vision`、`natural_language_processing`、`deep_learning_framework`、`robotics`、`embedded_iot`、`data_science` 中的哪一类。
2. 开源项目成长热度预测：根据 README 文本和仓库元数据预测项目成长热度等级 `low`、`medium`、`high`。

## 数据来源

数据来自 GitHub 官方 REST API，不爬取网页 HTML。

仓库搜索接口：

```text
https://api.github.com/search/repositories
```

README 获取接口：

```text
https://api.github.com/repos/{full_name}/readme
```

项目支持可选 `GITHUB_TOKEN`。没有 Token 时也能低速采集；如果配置了 Token，脚本会自动使用认证请求。

## 采集关键词

`crawler.py` 使用以下专业技术方向关键词搜索公开仓库：

```text
computer vision
image classification
object detection
semantic segmentation
natural language processing
large language model
transformer
deep learning
pytorch
tensorflow
robotics
ros
slam
embedded systems
stm32
esp32
arduino
fpga
iot
edge computing
signal processing
```

## 数据字段

`data/raw_repos.csv` 保存 GitHub 仓库元数据：

| 字段 | 说明 |
| --- | --- |
| repo_name | 仓库名称 |
| full_name | 仓库完整名称，格式为 owner/repo |
| description | 仓库描述 |
| language | 主要编程语言 |
| topics | GitHub topics 标签 |
| stargazers_count | stars 数量 |
| forks_count | forks 数量 |
| watchers_count | watchers 数量 |
| open_issues_count | open issues 数量 |
| size | 仓库大小 |
| created_at | 创建时间 |
| updated_at | 更新时间 |
| pushed_at | 最近 push 时间 |
| archived | 是否归档 |
| has_issues | 是否启用 issues |
| license | 开源许可证 |
| html_url | GitHub 页面地址 |
| api_url | GitHub API 地址 |
| query_keyword | 采集时使用的关键词 |
| crawl_time | 采集时间 |

`fetch_readme.py` 新增：

| 字段 | 说明 |
| --- | --- |
| readme_text | base64 解码后的 README 文本 |
| readme_length | README 字符长度 |

`clean_data.py` 新增：

| 字段 | 说明 |
| --- | --- |
| combined_text | repo_name、description、topics、language、readme_text 拼接文本 |
| project_category | 技术方向类别 |
| repo_age_days | 仓库创建至今的天数 |
| days_since_update | 距最近更新的天数 |
| repo_age_years | 仓库年龄，单位为年 |
| growth_score | 成长热度指标 |
| growth_popularity_level | 成长热度等级 |

## 技术方向标签构建方法

项目根据 `query_keyword` 和 `topics` 生成 `project_category` 标签。若一个样本可能匹配多个类别，优先使用 `query_keyword` 对应类别。

类别包括：

- `computer_vision`
- `natural_language_processing`
- `deep_learning_framework`
- `robotics`
- `embedded_iot`
- `data_science`

## TextCNN 模型结构

`train_textcnn.py` 使用 PyTorch 从零实现 TextCNN，不依赖 transformers 或预训练模型。

模型结构：

1. Tokenization
2. Vocabulary
3. Embedding 层
4. 多个 `Conv1d` 卷积层，kernel size 为 3、4、5
5. ReLU 激活
6. MaxPooling
7. Dropout
8. Fully Connected
9. Softmax 输出分类结果

训练过程中自动检测设备：有 CUDA 则使用 GPU，否则使用 CPU。

## 传统机器学习模型

`train_ml_models.py` 使用 TF-IDF 表示 `combined_text`，并训练对比以下模型：

- KNN
- Logistic Regression
- Linear SVM
- Random Forest

评价指标包括 Accuracy、Precision、Recall、F1-score。

## 成长热度指标

成长热度预测任务的目标标签为 `growth_popularity_level`。

成长热度指标定义：

```text
repo_age_years = repo_age_days / 365
growth_score = log1p(stargazers_count) / log1p(repo_age_days + 1)
```

然后根据 `growth_score` 三分位数生成：

```text
low / medium / high
```

成长热度模型输入特征包括：

```text
combined_text
repo_age_days
days_since_update
open_issues_count
size
archived
has_issues
```

为避免数据泄露，不直接使用 `stargazers_count` 和 `watchers_count` 作为输入特征。

## 项目结构

```text
AI_GitHub_Project_Intelligence/
├── README.md
├── requirements.txt
├── crawler.py
├── fetch_readme.py
├── clean_data.py
├── train_ml_models.py
├── train_textcnn.py
├── train_growth_model.py
├── evaluate.py
├── visualize.py
├── data/
│   ├── raw_repos.csv
│   ├── repos_with_readme.csv
│   └── clean_repos.csv
├── models/
│   ├── best_ml_model.joblib
│   ├── textcnn_model.pth
│   └── tokenizer.joblib
├── results/
│   ├── category_count.png
│   ├── language_distribution.png
│   ├── readme_length_distribution.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── textcnn_training_curve.png
│   ├── confusion_matrix_ml.png
│   ├── confusion_matrix_textcnn.png
│   ├── growth_score_distribution.png
│   ├── classification_report_ml.txt
│   ├── classification_report_textcnn.txt
│   └── wrong_cases.csv
└── screenshots/
    └── README_PLACEHOLDER.txt
```

## 环境配置

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 包含：

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
requests
joblib
torch
```

## GitHub Token 可选配置

Windows PowerShell：

```powershell
$env:GITHUB_TOKEN="your_github_token_here"
```

macOS / Linux：

```bash
export GITHUB_TOKEN="your_github_token_here"
```

配置后运行脚本时，`crawler.py` 和 `fetch_readme.py` 会自动使用 Token。

## 运行步骤

请在项目根目录 `AI_GitHub_Project_Intelligence/` 下按顺序运行：

```bash
python -m pip install -r requirements.txt
python crawler.py
python fetch_readme.py
python clean_data.py
python train_ml_models.py
python train_textcnn.py
python train_growth_model.py
python evaluate.py
python visualize.py
```

## 结果文件说明

| 文件 | 说明 |
| --- | --- |
| data/raw_repos.csv | GitHub API 采集的仓库元数据 |
| data/repos_with_readme.csv | 增加 README 文本后的数据 |
| data/clean_repos.csv | 清洗、标签构建和特征工程后的数据 |
| models/best_ml_model.joblib | 最佳传统机器学习模型 |
| models/textcnn_model.pth | TextCNN 模型参数 |
| models/tokenizer.joblib | TextCNN 词表和标签映射 |
| results/model_comparison.csv | 传统模型与 TextCNN 指标对比 |
| results/model_comparison.png | 模型性能对比柱状图 |
| results/classification_report_ml.txt | 最佳传统机器学习模型分类报告 |
| results/classification_report_textcnn.txt | TextCNN 分类报告 |
| results/confusion_matrix_ml.png | 传统机器学习混淆矩阵 |
| results/confusion_matrix_textcnn.png | TextCNN 混淆矩阵 |
| results/wrong_cases.csv | 传统机器学习模型错误案例 |
| results/growth_model_comparison.csv | 成长热度预测模型对比 |
| results/growth_classification_report.txt | 成长热度预测分类报告 |
| results/growth_confusion_matrix.png | 成长热度预测混淆矩阵 |
| results/category_count.png | 技术方向类别分布图 |
| results/language_distribution.png | 编程语言分布图 |
| results/readme_length_distribution.png | README 长度分布图 |
| results/growth_score_distribution.png | 成长热度分布图 |
| results/evaluation_summary.txt | 实验结果摘要 |
| results/text_feature_tsne_balanced.png | README 文本特征二维分布图 |
| results/prediction_cases_compact.csv | 典型样本预测案例表，用于论文定性分析 |
| results/prediction_cases_compact.png | 紧凑版典型样本预测案例图片 |
| results/category_keywords_tfidf.png | 各技术方向 Top TF-IDF 关键词对比图 |
| results/category_keywords_tfidf.csv | 各类别 Top TF-IDF 关键词表 |
| results/paper_table_ready_cases.csv | 适合复制到 Word 的中文典型预测案例表 |
| results/textcnn_parameter_table.csv | TextCNN 参数设置表 |
| results/textcnn_epoch_summary.csv | TextCNN 训练轮次变化表 |
| results/paper_insert_text.txt | 可直接复制到论文中的补充文字 |

## 论文展示增强材料

在不重新采集数据、不重新训练 TextCNN 的前提下，可以运行以下脚本生成课程论文展示增强材料：

```bash
python plot_category_keywords.py
python export_prediction_cases_compact.py
python export_paper_tables.py
python visualize.py
```

增强材料说明：

- `results/category_keywords_tfidf.png`：各技术方向关键词对比图，适合作为论文“特征工程”或“实验结果可视化”的核心展示图。
- `results/category_keywords_tfidf.csv`：各类别 Top TF-IDF 关键词，可用于补充表格或核对图中关键词。
- `results/paper_table_ready_cases.csv`：适合复制到 Word 的典型预测案例表，建议放在“错误案例分析”或“定性分析”小节。
- `results/text_feature_tsne_balanced.png`：用于展示 README 文本特征在二维空间中的分布与类别交叠情况。
- `results/prediction_cases_compact.csv`：用于论文中的典型样本定性分析。
- `results/prediction_cases_compact.png`：用于论文插图或作为表格图片参考。
- `results/textcnn_parameter_table.csv`：用于论文“参数设置”表格。
- `results/textcnn_epoch_summary.csv`：用于论文“TextCNN 训练过程分析”表格。
- `results/paper_insert_text.txt`：用于补充 TextCNN 结构解释、训练轮次分析、成长热度预测讨论和 t-SNE 图分析。

## 截图建议

课程论文中建议截图：

1. `python crawler.py` 终端输出，展示关键词、页数、累计采集数量。
2. `python fetch_readme.py` 终端输出，展示 README 成功和失败数量。
3. `python clean_data.py` 终端输出，展示清洗前后样本数、技术方向类别数量、成长热度类别数量。
4. `python train_ml_models.py` 终端输出，展示传统模型对比指标。
5. `python train_textcnn.py` 终端输出，展示每轮 train_loss、train_accuracy、test_accuracy。
6. `python train_growth_model.py` 终端输出，展示成长热度预测模型对比。
7. `results/` 中的模型对比图、训练曲线、混淆矩阵和数据分布图。
8. `results/wrong_cases.csv` 中的错误案例，用于误差分析。

## Rate Limit 说明

如果 GitHub API 出现 rate limit：

- 等待脚本自动 sleep 后继续运行；
- 配置 `GITHUB_TOKEN` 后重新运行；
- 稍后再运行采集脚本；
- 适当减少 `MAX_PAGES_PER_KEYWORD` 或增加请求间隔。

本项目只使用 GitHub 官方 REST API，不进行网页 HTML 爬取，也不绕过登录、验证码或反爬机制。
