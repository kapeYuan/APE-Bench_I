[English](#english-version) | [中文](#chinese-version)
<a name="english-version"></a>

# 3.2 Data Handling: Tasks and Format

This section describes how APE-Bench I tasks are structured, where the data comes from, and how it's handled by the `src/apebench/data/` modules.

## Task Format

As specified in the APE-Bench I paper (Section 3.1), each task in the benchmark is a triplet: `(Instruction, PreFile, Patch)`.

*   **`Instruction`**: A natural language string describing the intended modification to a Lean file. This serves as the main prompt for the LLM being evaluated.
    *   *Example*: "Refactor the proof of `theorem_xyz` to use `lemma_abc`." or "Add a new definition `new_function` with the following properties..."
*   **`PreFile`**: A string containing the complete Lean source code of the target file *before* the edit. This provides the full context for the LLM.
*   **`Patch`**: A string in the unified diff format that encodes the ground-truth edit. This patch, when applied to `PreFile`, should result in the desired post-edit state of the file.
    *   This is used as the reference for evaluating LLM-generated patches, although direct diff matching is not the primary success metric (semantic correctness is key).

Additional metadata associated with each task in the test set includes:
*   **Task ID**: A unique identifier for the task.
*   **Commit SHA**: The Mathlib4 commit from which the task was derived.
*   **File Path**: The path to the specific Lean file within the Mathlib commit.
*   **Task Category**: One of `Feature`, `Refactor`, or `Bug Fix` (as defined in paper Section 3.3).
*   **Difficulty Level**: One of `Easy`, `Medium`, or `Hard` (as defined in paper Section 3.3).

## Data Source

The APE-Bench I dataset is hosted on Hugging Face:
*   **URL**: [https://huggingface.co/datasets/HuajianXin/APE-Bench_I](https://huggingface.co/datasets/HuajianXin/APE-Bench_I)

During setup, you must clone this dataset into the `datasets/` directory within your project. The primary test dataset file is named `ape_bench1_test.parquet`.

## Data Handling in `src/apebench/data/`

The modules within `src/apebench/data/` are responsible for:

*   **Loading Tasks**: Reading the benchmark data files (from `datasets/`) into memory, supporting both JSONL and Parquet formats.
*   **Parsing**: Extracting the `Instruction`, `PreFile`, `Patch`, and other metadata for each task.
*   **Data Representation**: Converting the raw data into Python objects for easier use throughout the application.
*   **Filtering/Selection**: Providing utilities to classify specific tasks based on criteria like ID, category, or difficulty.

---
<a name="chinese-version"></a>

## 中文翻译 (Chinese Translation)

# 3.2 数据处理：任务与格式

本节描述 APE-Bench I 任务的结构、数据来源以及 `src/apebench/data/` 模块如何处理这些数据。

## 任务格式

正如 APE-Bench I 论文（第 3.1 节）所明确指出的，基准测试中的每个任务都是一个三元组：`(Instruction, PreFile, Patch)`。

*   **`Instruction` (指令)**：一个自然语言字符串，描述对 Lean 文件的预期修改。这是被评估 LLM 的主要提示。
    *   *示例*："将 `theorem_xyz` 的证明重构为使用 `lemma_abc`。"或"添加一个具有以下属性的新定义 `new_function`..."
*   **`PreFile` (修改前文件)**：一个包含编辑前目标文件完整 Lean 源代码的字符串。这为 LLM 提供了完整的上下文。
*   **`Patch` (补丁)**：一个统一差异格式的字符串，编码了真实的编辑。当此补丁应用于 `PreFile` 时，应产生文件所需的编辑后状态。
    *   这被用作评估 LLM 生成补丁的参考，尽管直接的差异匹配不是主要的成功指标（语义正确性是关键）。

测试集中与每个任务相关的其他元数据包括：
*   **Task ID (任务 ID)**：任务的唯一标识符。
*   **Commit SHA (提交 SHA)**：任务来源的 Mathlib4 提交的 SHA 值。
*   **File Path (文件路径)**：Mathlib 提交中特定 Lean 文件的路径。
*   **Task Category (任务类别)**：`Feature` (功能)、`Refactor` (重构)或`Bug Fix` (错误修复)之一（根据论文第 3.3 节定义）。
*   **Difficulty Level (难度级别)**：`Easy` (简单)、`Medium` (中等)或`Hard` (困难)之一（根据论文第 3.3 节定义）。

## 数据来源

APE-Bench I 数据集托管在 Hugging Face 上：
*   **URL**: [https://huggingface.co/datasets/HuajianXin/APE-Bench_I](https://huggingface.co/datasets/HuajianXin/APE-Bench_I)

在设置过程中，您必须将此数据集克隆到项目中的 `datasets/` 目录。主要的测试数据集文件名为 `ape_bench1_test.parquet`。

## `src/apebench/data/` 中的数据处理

`src/apebench/data/` 中的模块负责：

*   **加载任务**：从 `datasets/` 内存中读取基准测试数据文件，支持 JSONL 和 Parquet 格式。
*   **解析**：为每个任务提取 `Instruction`、`PreFile`、`Patch` 和其他元数据。
*   **数据表示**：将原始数据转换为 Python 对象，以便在整个应用程序中更轻松地使用。
*   **筛选/选择**：提供根据 ID、类别或难度等标准分类。

---

下一节: [LLM 推理与 DiffRepair](./03_3_apebench_inference.md) 