[English](#english-version) | [中文](#chinese-version)
<a name="english-version"></a>

# 1. Setup and Installation

This section guides you through setting up the APE-Bench I project environment.

## Prerequisites

*   **Operating System**: Linux is recommended. 
*   **Git**: For cloning the repository and Mathlib4.
*   **Python**: Version 3.9 or higher.
*   **Lean 4 and Lake**: Required for Eleanstic setup and local Lean file verification. You must separately install Lean 4 and Elan (Lean version manager) and add them to your PATH, see the official [Lean installation guide](https://leanprover-community.github.io/get_started.html).
*   **Disk space**: Eleanstic's preprocessed data for Mathlib requires approximately 1.1 TB of storage space for the complete benchmark. The APE-Bench I dataset itself is smaller (~50 MB).

## 1. Clone the Repository

Clone this project repository to your local machine:

```bash
git clone https://github.com/xinhjBrant/APE-Bench_I
cd APE-Bench_I
```

## 2. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 3. Download the APE-Bench I Dataset

The APE-Bench I dataset contains the tasks (Instruction, PreFile, Patch) used for evaluation. It is hosted on Hugging Face.

As specified in the project's `README.md`:

```bash
# Ensure you are in the root directory of the cloned APE-Bench_I project
mkdir datasets
git clone https://huggingface.co/datasets/HuajianXin/APE-Bench_I datasets
```

This will download the dataset into the `datasets` directory within your project. The primary test dataset file is named `ape_bench1_test.parquet`.

## 4. Setting up Eleanstic

Eleanstic is crucial for syntactic verification. It requires a one-time setup process to download and preprocess the versions of Mathlib4 relevant to the benchmark tasks. This step is time-consuming and resource-intensive.

**Steps for Eleanstic Setup:**

1.  **Mathlib4 Clone**: Clone the `leanprover-community/mathlib4` repository locally. Eleanstic will access this repository to check out specific commits.

    ```bash
    git clone https://github.com/leanprover-community/mathlib4.git
    ```

2.  **Configuration**: Configure Eleanstic by editing `src/eleanstic/config.yaml`. Key parameters include:
    ```yaml
    # Example (src/eleanstic/config.yaml)
    paths:
      mathlib_repo: "/path/to/your/mathlib4_clone"  # Absolute path to your mathlib4 clone
      workspace_root: "verify_database"  # Base directory for Eleanstic data
      worktree_dir: "worktrees"  # For temporary Git worktrees
      storage_dir: "storage"     # For content-addressable storage
      cache_dir: "cache"         # For Lake cache
      log_dir: "logs"            # For log files
      verify_results_dir: "./verify_results"  # Results output

    concurrency:
      max_workers: 128  # Adjust based on your system's capability
      max_concurrent_file_storage: 8
      max_concurrent_lean_verifications: 64

    storage:
      hash_algorithm: "sha256"  # Hashing algorithm for content-addressable storage
      remove_worktree_after_build: true  # Remove temporary worktrees after building
    ```

3.  **Preprocessing Commits**: Run the Eleanstic build command to process the Mathlib commits used in APE-Bench I. Eleanstic preprocesses all Mathlib commits referenced in the APE-Bench dataset file (e.g., `datasets/ape_bench1_test.parquet`).

    The `eleanstic.main build` command takes the APE-Bench `.parquet` dataset file as its `--input_file`. Eleanstic parses this file to identify the commit hashes required for preprocessing.

    ```bash
    python -m src.eleanstic.main \
        --config src/eleanstic/config.yaml \
        --input_file datasets/ape_bench1_test.parquet \
        --commit_id_key commit_hash \
        build
    ```

    **Alternative: Using a custom list of commits (JSONL):**
    ```bash
    python -m src.eleanstic.main \
        --config src/eleanstic/config.yaml \
        --input_file /path/to/your_commits_to_build.jsonl \
        --commit_id_key commit_hash \
        --max_workers <num_processes> \
        build
    ```
    
    The input file (whether `.parquet` or `.jsonl`) must contain the commit identifiers under the key specified by `--commit_id_key`. If using a JSONL file and specific Lean toolchains are needed per commit, include them under a `toolchain_content` key.

    This build process will:
    - Check out each commit from your Mathlib repository
    - Build it with Lake
    - Store the build artifacts in the content-addressable storage
    - Create snapshots for restoration during verification

**Note**: This preprocessing step is resource-intensive but only needs to be done once. The paper demonstrates storage reduction from 15.6 TB to 1.1 TB for thousands of commits through Eleanstic's deduplication technology.

## 5. API Keys (for running LLM inference)

If you plan to run inference with LLMs (OpenAI, Anthropic, Google models), you must set up your API keys. The repository includes an example file `src/apebench/inference/utils/api_keys.example.py` with the structure needed.

**Steps to configure API keys:**

1. Copy the example file to create your own API keys file:
   ```bash
   cp src/apebench/inference/utils/api_keys.example.py src/apebench/inference/utils/api_keys.py
   ```

2. Edit `api_keys.py` with your actual API keys. The file is already in `.gitignore` to prevent accidental commits of sensitive information.

Example structure within `api_keys.py`:
```python
# OpenAI API credentials (GPT models)
openai_api_key = "your-openai-api-key" # Replace with your actual key
openai_base_url = "https://api.openai.com/v1"  # Or your Azure OpenAI endpoint

# Anthropic API credentials (Claude models)
aws_claude_api_key = "your-anthropic-api-key" # Replace with your actual key
aws_claude_base_url = "https://api.anthropic.com"  # Or your AWS Claude endpoint

# DeepSeek models
volces_api_key = "your-deepseek-api-key" # Replace with your actual key
volces_base_url = "https://api.deepseek.com"

# Google API credentials
google_api_key = "your-google-api-key" # Replace with your actual key
google_base_url = "https://generativelanguage.googleapis.com"

# Add additional API credentials as needed for other models
```

Ensure this file has your keys populated before running any inference tasks that require these LLMs.

After completing these steps, your environment is ready for running experiments with the APE-Bench I framework.

---

Next: [Project Structure](./02_project_structure.md)

<a name="chinese-version"></a>

## 中文翻译 (Chinese Translation)

# 1. 安装与设置

本节指导您完成 APE-Bench I 项目环境的设置。

## 先决条件

*   **操作系统**：推荐使用 Linux。
*   **Git**：用于克隆代码仓库和 Mathlib4。
*   **Python**：版本 3.9 或更高。
*   **Lean 4 和 Lake**：Eleanstic 设置和本地 Lean 文件验证所必需。您必须单独安装 Lean 4 和 Elan（Lean 版本管理器）并将其添加到 PATH 中，请遵循 [Lean 社区安装指南](https://leanprover-community.github.io/get_started.html)。
*   **磁盘空间**：Eleanstic 为 Mathlib 预处理的数据需要约 1.1 TB 的存储空间用于完整基准测试。APE-Bench I 数据集本身较小（约 50 MB）。

## 1. 克隆代码仓库

将此项目代码仓库克隆到您的本地计算机：

```bash
git clone https://github.com/xinhjBrant/APE-Bench_I
cd APE-Bench_I
```

## 2. Python 依赖

安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

## 3. 下载 APE-Bench I 数据集

APE-Bench I 数据集包含用于评估的任务（指令、修改前文件、补丁）。它托管在 Hugging Face 上。

正如项目 `README.md` 中所指定的：

```bash
# 确保您位于克隆的 APE-Bench_I 项目的根目录中
mkdir datasets
git clone https://huggingface.co/datasets/HuajianXin/APE-Bench_I datasets
```

这会将数据集下载到您项目中的 `datasets` 目录。主要的测试数据集文件名为 `ape_bench1_test.parquet`。

## 4. 设置 Eleanstic

Eleanstic 对于语法验证至关重要。它需要一次性设置过程来下载和预处理与基准测试任务相关的 Mathlib4 版本。此步骤耗时且资源密集。

**Eleanstic 设置步骤：**

1.  **Mathlib4 克隆**：在本地克隆 `leanprover-community/mathlib4` 代码仓库。Eleanstic 将访问此仓库以检出特定的提交。

    ```bash
    git clone https://github.com/leanprover-community/mathlib4.git
    ```

2.  **配置**：通过编辑 `src/eleanstic/config.yaml` 来配置 Eleanstic。关键参数包括：
    ```yaml
    # 示例 (src/eleanstic/config.yaml)
    paths:
      mathlib_repo: "/path/to/your/mathlib4_clone" # 指向您的 mathlib4 克隆的绝对路径
      workspace_root: "verify_database"  # Eleanstic 数据的基础目录
      worktree_dir: "worktrees"  # 用于临时 Git 工作区
      storage_dir: "storage"     # 用于内容寻址存储
      cache_dir: "cache"         # 用于 Lake 缓存
      log_dir: "logs"            # 用于日志文件
      verify_results_dir: "./verify_results"  # 结果输出

    concurrency:
      max_workers: 128  # 根据您的系统能力调整
      max_concurrent_file_storage: 8
      max_concurrent_lean_verifications: 64

    storage:
      hash_algorithm: "sha256"  # 内容寻址存储的哈希算法
      remove_worktree_after_build: true  # 构建后移除临时工作区
    ```

3.  **预处理提交**：运行 Eleanstic 构建命令来处理 APE-Bench I 中使用的 Mathlib 提交。Eleanstic 预处理 APE-Bench 数据集文件（例如 `datasets/ape_bench1_test.parquet`）中引用的所有 Mathlib 提交。

    `eleanstic.main build` 命令将 APE-Bench `.parquet` 数据集文件作为其 `--input_file`。Eleanstic 解析此文件以识别预处理所需的提交哈希值。

    ```bash
    python -m src.eleanstic.main \
        --config src/eleanstic/config.yaml \
        --input_file datasets/ape_bench1_test.parquet \
        --commit_id_key commit_hash \
        build
    ```

    **替代方案：使用自定义提交列表 (JSONL)：**
    ```bash
    python -m src.eleanstic.main \
        --config src/eleanstic/config.yaml \
        --input_file /path/to/your_commits_to_build.jsonl \
        --commit_id_key commit_hash \
        --max_workers <进程数> \
        build
    ```
    
    输入文件（无论是 `.parquet` 还是 `.jsonl`）必须在 `--commit_id_key` 指定的键下包含提交标识符。如果使用 JSONL 文件且每个提交需要特定的 Lean 工具链，请将其包含在 `toolchain_content` 键下。

    此构建过程将：
    - 从您的 Mathlib 代码仓库中检出每个提交
    - 使用 Lake 构建它
    - 将构建产物存储在内容寻址存储中
    - 创建快照以便在验证期间恢复

**注意**：此预处理步骤资源密集，但只需要执行一次。论文证明，通过 Eleanstic 的去重技术，数千个提交的存储空间从 15.6 TB 减少到 1.1 TB。

## 5. API 密钥 (用于运行 LLM 推理)

如果您计划使用 LLM（OpenAI、Anthropic、Google 模型）进行推理，您必须设置 API 密钥。代码仓库包含一个示例文件 `src/apebench/inference/utils/api_keys.example.py`，其中包含所需的结构。

**配置 API 密钥的步骤：**

1. 复制示例文件以创建您自己的 API 密钥文件：
   ```bash
   cp src/apebench/inference/utils/api_keys.example.py src/apebench/inference/utils/api_keys.py
   ```

2. 使用您的实际 API 密钥编辑 `api_keys.py`。该文件已添加到 `.gitignore` 中，以防止意外提交敏感信息。

`api_keys.py` 中的示例结构：
```python
# OpenAI API 凭据 (GPT 模型)
openai_api_key = "your-openai-api-key" # 替换为您的实际密钥
openai_base_url = "https://api.openai.com/v1"  # 或您的 Azure OpenAI 端点

# Anthropic API 凭据 (Claude 模型)
aws_claude_api_key = "your-anthropic-api-key" # 替换为您的实际密钥
aws_claude_base_url = "https://api.anthropic.com"  # 或您的 AWS Claude 端点

# DeepSeek 模型
volces_api_key = "your-deepseek-api-key" # 替换为您的实际密钥
volces_base_url = "https://api.deepseek.com"

# Google API 凭据
google_api_key = "your-google-api-key" # 替换为您的实际密钥
google_base_url = "https://generativelanguage.googleapis.com"

# 根据需要为其他模型添加额外的 API 凭据
```

在运行任何需要这些 LLM 的推理任务之前，请确保此文件已填充您的密钥。

完成这些步骤后，您的环境已准备好使用 APE-Bench I 框架运行实验。

---

下一节: [项目结构](./02_project_structure.md)