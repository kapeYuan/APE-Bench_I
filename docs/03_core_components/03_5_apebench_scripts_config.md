[English](#english-version) | [中文](#chinese-version)
<a name="english-version"></a>

# 3.5 Scripts and Configuration

This section covers the scripts and configuration files that orchestrate the APE-Bench evaluation workflows.

## Scripts (`src/apebench/scripts/`)

The `src/apebench/scripts/` directory typically contains Python scripts that serve as high-level entry points for various stages of the APE-Bench I workflow.

**Common Functions of Scripts:**

*   **Running Full Experiments**: Scripts to orchestrate an end-to-end evaluation for one or more LLMs. This might involve:
    *   Loading tasks from the dataset.
    *   Calling the inference modules (`src/apebench/inference/`) to generate patches for all tasks.
    *   Invoking the evaluation pipeline (`src/apebench/evaluation_pipelines/`) to perform syntactic and semantic checks.
    *   Saving raw results and aggregated metrics.
*   **Data Preprocessing/Analysis**: Scripts for analyzing the APE-Bench I dataset itself, or for preprocessing data before an experiment.
*   **Result Aggregation and Reporting**: Scripts to collect results from multiple partial runs, compute final metrics (like those in the paper's tables and figures), and generate reports or visualizations.
    *   This might use `src/apebench/evaluation_pipelines/gather_results.py` internally.
*   **Targeted Evaluations**: Scripts for running specific parts of the pipeline, e.g., only running inference for a new model, or only re-evaluating existing patches with a new semantic judge.

**Usage:**

These scripts are generally designed to be run from the command line. They would parse command-line arguments to specify things like:
*   Which LLM(s) to evaluate.
*   Paths to input data and output directories.
*   Configuration files to use.
*   Specific task IDs or categories to focus on.

## Configuration Files

Configuration files allow for customizing the behavior of the APE-Bench framework without modifying the source code directly.

### 1. Eleanstic Configuration (`src/eleanstic/config.yaml`)

*   **Purpose**: Configures the Eleanstic service.
*   **Key Settings**: As detailed in the [Eleanstic documentation](./04_1_eleanstic.md):
    *   `mathlib_repo_path`: Path to your local Mathlib4 clone.
    *   `cas_store_path`: Location for Eleanstic's Content-Addressable Store.
    *   `snapshots_path`: Location for Eleanstic's commit snapshots.
    *   Parameters for concurrency, logging, etc.
*   **Importance**: Must be correctly set up before Eleanstic can be used, especially for the initial preprocessing of Mathlib commits.

### 2. APE-Bench Configuration (primarily in `src/apebench/config/`)

This directory likely contains configuration files (e.g., YAML, JSON, or Python modules) for various aspects of the APE-Bench experiments.

*   **Model Configurations**: 
    *   API keys (or paths to key files).
    *   Model names/identifiers as used in APIs (e.g., `gpt-4o`, `claude-3-sonnet-20240229`).
    *   Default generation parameters (temperature, max tokens, top_p) for each model.
    *   API endpoint URLs if not standard.
*   **Path Configurations**: 
    *   Paths to the APE-Bench I dataset (`datasets/`).
    *   Default directories for saving LLM-generated patches, evaluation results, logs, and analysis outputs.
*   **Experiment Parameters**: 
    *   Number of samples to generate per task ($n$ for pass@k).
    *   Parameters for `DiffRepair` (e.g., matching thresholds).
    *   Settings for the LLM-as-a-Judge (e.g., which model to use as judge, judge-specific prompting parameters).
*   **Feature Flags**: Flags to enable/disable certain parts of the pipeline (e.g., skip syntactic check, force re-generation of patches).

## Secondary Development

*   **Scripts**:
    *   Develop new scripts for novel experimental workflows or more detailed analyses (e.g., generating specific plots, performing statistical tests on results).
    *   Improve the command-line interface and modularity of existing scripts.
*   **Configuration**:
    *   Refine the structure of configuration files for better organization or to support more complex experimental designs (e.g., using hierarchical configurations with tools like Hydra).
    *   Add validation for configuration parameters (e.g., using `pydantic` as listed in `requirements.txt`) to catch errors early.
    *   Standardize how different modules access configuration settings.

Effectively using and managing scripts and configurations is key to running reproducible experiments and extending the APE-Bench I framework.

---

Next: [Troubleshooting](./04_troubleshooting.md)

<a name="chinese-version"></a>

## 中文翻译 (Chinese Translation)

# 3.5 脚本与配置

本节介绍协调 APE-Bench 评估工作流的脚本和配置文件。

## 脚本 (`src/apebench/scripts/`)

`src/apebench/scripts/` 目录通常包含作为 APE-Bench I 工作流程各个阶段高级入口点的 Python 脚本。

**脚本的常见功能：**

*   **运行完整实验**：用于为一个或多个 LLM 编排端到端评估的脚本。这可能涉及：
    *   从数据集中加载任务。
    *   调用推理模块 (`src/apebench/inference/`) 为所有任务生成补丁。
    *   调用评估流程 (`src/apebench/evaluation_pipelines/`) 执行语法和语义检查。
    *   保存原始结果和聚合指标。
*   **数据预处理/分析**：用于分析 APE-Bench I 数据集本身，或在实验前预处理数据的脚本。
*   **结果聚合和报告**：用于从多个部分运行中收集结果，计算最终指标（如论文表格和图中的指标），并生成报告或可视化的脚本。
    *   这可能在内部使用 `src/apebench/evaluation_pipelines/gather_results.py`。
*   **有针对性的评估**：用于运行流程特定部分的脚本，例如，仅为新模型运行推理，或仅使用新的语义裁判重新评估现有补丁。

**用法：**

这些脚本通常设计为从命令行运行。它们会解析命令行参数以指定诸如以下内容：
*   要评估的 LLM。
*   输入数据和输出目录的路径。
*   要使用的配置文件。
*   要关注的特定任务 ID 或类别。

## 配置文件

配置文件允许在不直接修改源代码的情况下自定义 APE-Bench 框架的行为。

### 1. Eleanstic 配置 (`src/eleanstic/config.yaml`)

*   **目的**：配置 Eleanstic 服务。
*   **关键设置**：如 [Eleanstic 文档](./04_1_eleanstic.md) 中所述：
    *   `mathlib_repo_path`：指向您的本地 Mathlib4 克隆的路径。
    *   `cas_store_path`：Eleanstic 内容寻址存储的位置。
    *   `snapshots_path`：Eleanstic 提交快照的位置。
    *   并发、日志记录等参数。
*   **重要性**：在使用 Eleanstic 之前必须正确设置，尤其是在对 Mathlib 提交进行初始预处理时。

### 2. APE-Bench 配置 (主要在 `src/apebench/config/` 中)

此目录可能包含 APE-Bench 实验各个方面的配置文件（例如 YAML、JSON 或 Python 模块）。

*   **模型配置**：
    *   API 密钥（或密钥文件路径）。
    *   API 中使用的模型名称/标识符（例如 `gpt-4o`、`claude-3-sonnet-20240229`）。
    *   每个模型的默认生成参数（温度、最大令牌数、top_p）。
    *   如果不是标准 API，则为 API 端点 URL。
*   **路径配置**：
    *   指向 APE-Bench I 数据集的路径 (`datasets/`)。
    *   用于保存 LLM 生成的补丁、评估结果、日志和分析输出的默认目录。
*   **实验参数**：
    *   每个任务生成的样本数（pass@k 中的 $n$）。
    *   `DiffRepair` 的参数（例如匹配阈值）。
    *   作为裁判的 LLM 的设置（例如使用哪个模型作为裁判，裁判特定的提示参数）。
*   **功能标志**：用于启用/禁用流程某些部分的标志（例如跳过语法检查，强制重新生成补丁）。

---

下一节: [故障排除](./04_troubleshooting.md)