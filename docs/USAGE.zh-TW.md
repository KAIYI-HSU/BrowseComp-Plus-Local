# fireworks_eval 使用說明（繁體中文）

一套建立在 **BrowseComp-Plus** benchmark 上、**100% 自動化**的 Deep-Research Agent 評估
pipeline。本文件帶你從零開始跑通評估、切換模型 / 提示 / 資料來源、評估你自己的 agent，
並提供**遷移到 HPC Cluster（含離線環境）**所需的完整準備清單與步驟。

> 機器無關的設計：所有 LLM 呼叫走 OpenAI 相容端點（預設 **Fireworks**，可換供應商）；
> 檢索與 embedding 在**本機**執行（Qwen3-Embedding-0.6B + FAISS）。

---

## 目錄
1. [5 分鐘快速開始](#1-5-分鐘快速開始)
2. [環境準備](#2-環境準備)
3. [資料集與索引準備](#3-資料集與索引準備)
4. [執行你的第一個實驗](#4-執行你的第一個實驗)
5. [切換模型（逐階段／換供應商）](#5-切換模型)
6. [切換資料集／資料來源](#6-切換資料集資料來源)
7. [切換 Prompt](#7-切換-prompt)
8. [評估你自己的 Deep-Research Agent](#8-評估你自己的-agent)
9. [回歸對照：檢驗提升或副作用](#9-回歸對照)
10. [分領域（情境）評估](#10-分領域評估)
11. [輸出與指標說明](#11-輸出與指標說明)
12. [E2E Smoke Test](#12-e2e-smoke-test)
13. [自動化測試](#13-自動化測試)
14. [🚀 遷移到 HPC Cluster（含離線準備）](#14-遷移到-hpc-cluster)
15. [參數速查表](#15-參數速查表)
16. [常見問題排解](#16-常見問題排解)

---

## 1. 5 分鐘快速開始

> 前提：已在專案根目錄、`.env` 內有 `FIREWORKS_API_KEY=...`、且 `indexes/qwen3-embedding-0.6b/`
> 與 `data/browsecomp_plus_decrypted.jsonl` 已就緒（見 [§3](#3-資料集與索引準備)）。

```bash
# 啟用環境（uv 管理）
source .venv/bin/activate           # 或在每個指令前加上 .venv/bin/python

# E2E smoke test：抽 3 題跑完整 pipeline 並驗證輸出格式與合理性
python -m fireworks_eval.smoke_test
```

預期看到 `SMOKE TEST: PASS`，並在 `reports/e2e_smoke_claude/report.md` 產生報表。
第一次執行會：建立精簡版 ground-truth 快取（約 1 分鐘）、啟動本機檢索服務（首次載入
模型＋語料約 1–2 分鐘）、跑 3 題 agent ＋ judge ＋ 報表。

---

## 2. 環境準備

### 2.1 Python 與 uv
本專案以 **uv** 管理虛擬環境（Python 3.10）。

```bash
# 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在專案根目錄建立／同步環境（會依 pyproject.toml + uv.lock 安裝套件）
uv sync
source .venv/bin/activate
```

> ⚠️ 本專案的 `.venv` **沒有 pip**。要額外裝套件請用 `uv pip install <套件>`，
> 不要用 `.venv/bin/pip`。

pipeline 額外用到的套件（多半已隨環境安裝）：`openai`、`httpx`、`pyyaml`、`tenacity`、
`fastapi`、`uvicorn`、`python-dotenv`、`faiss-cpu`、`torch`、`transformers`、`tevatron`。
若缺少：`uv pip install openai httpx pyyaml tenacity fastapi uvicorn python-dotenv`。

### 2.2 API 金鑰（`.env`）
在專案根目錄建立 `.env`：

```dotenv
FIREWORKS_API_KEY=fw_xxxxxxxxxxxxxxxx
```

之後若改用其他供應商，只要新增對應金鑰（例如 `OPENAI_API_KEY=...`）並在 config 內把
`api_key_env` 指到該變數即可（見 [§5](#5-切換模型)）。

### 2.3（選用）測試框架
要跑單元測試需安裝 pytest：`uv pip install pytest pytest-cov`。

---

## 3. 資料集與索引準備

評估需要三類本機檔案。**在有網路的機器上準備一次即可**（之後可整包搬到離線環境）。

| 類別 | 路徑 | 取得方式 | 大小（約） |
|---|---|---|---|
| 解密後資料集 | `data/browsecomp_plus_decrypted.jsonl` | `decrypt_dataset.py`（見下） | ~2 GB |
| 查詢 TSV（選用） | `topics-qrels/queries.tsv` | 同上 `--generate-tsv` | 小 |
| Relevance 標註 | `topics-qrels/qrel_evidence.txt`、`qrel_golds.txt` | 已隨 repo | 小 |
| 稠密檢索索引 | `indexes/qwen3-embedding-0.6b/corpus.shard*.pkl` | `download_indexes.sh` | ~412 MB |
| 文件語料（執行期） | HuggingFace 快取 `Tevatron/browsecomp-plus-corpus` | 執行期自動載入 | ~數 GB |

### 3.1 下載並解密資料集（產生 830 題的 ground truth）
```bash
# 需先登入 HuggingFace（資料集為非公開／需授權時）
huggingface-cli login        # 或設定環境變數 HF_TOKEN=hf_xxx

python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```
此步驟下載 `Tevatron/browsecomp-plus`（test split）並在本機解密，輸出含
`query_id / query / answer / gold_docs / evidence_docs / negative_docs` 的 JSONL。

### 3.2 下載預建索引（Qwen3-Embedding-0.6B）
```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset --include="qwen3-embedding-0.6b/*" --local-dir ./indexes
```
若要更強的檢索器，可改 `--include="qwen3-embedding-4b/*"` 或 `qwen3-embedding-8b/*`
（檔案更大、需更多記憶體；建議在 GPU 環境使用）。`download_indexes.sh` 內含四種索引的指令。

### 3.3 語料（自動）
檢索服務啟動時會以 `load_dataset("Tevatron/browsecomp-plus-corpus")` 載入約 10 萬篇文件
（首次自動下載並快取到 HuggingFace cache）。離線環境需預先快取，見 [§14](#14-遷移到-hpc-cluster)。

---

## 4. 執行你的第一個實驗

一個實驗 = 一個 config 檔（YAML/JSON）。最完整的範例與逐欄註解在
`configs/experiment.example.yaml`。最小可用 config：

```yaml
experiment_name: my_run
dataset:
  query_ids: ["769", "770", "771"]   # 或用 sample: 50 隨機抽樣
agent:
  model: accounts/fireworks/models/gpt-oss-120b
  query_template: QUERY_TEMPLATE_NO_GET_DOCUMENT
judge:
  model: accounts/fireworks/models/gpt-oss-120b
```

執行：
```bash
python -m fireworks_eval.run_experiment --config configs/experiment.example.yaml

# 常用覆寫（不改檔）：
python -m fireworks_eval.run_experiment --config configs/experiment.example.yaml \
  --experiment-name my_run_v2 --sample 50 --tagging
```
結束後會印出 Accuracy / Recall / 搜尋次數等，並輸出到 `runs/<name>/`、`evals/<name>/`、
`reports/<name>/`。重跑會自動跳過已完成的題目（`--force-agent` / `--force-judge` 可強制重跑）。

---

## 5. 切換模型

模型在 config 內分兩個階段、可獨立切換：

```yaml
agent:
  model: accounts/fireworks/models/gpt-oss-120b   # 受測 Deep-Research agent 的 LLM
  base_url: https://api.fireworks.ai/inference/v1
  api_key_env: FIREWORKS_API_KEY
  reasoning_effort: high                          # gpt-oss 支援 low|medium|high
judge:
  model: accounts/fireworks/models/gpt-oss-120b   # LLM-as-judge 的 LLM（可與 agent 不同）
  base_url: https://api.fireworks.ai/inference/v1
  api_key_env: FIREWORKS_API_KEY
```

### 5.1 換成另一個 Fireworks 模型
把 `model` 換成其它 Fireworks 模型 ID（例如 `accounts/fireworks/models/gpt-oss-20b`）。
完整模型清單請見 Fireworks 模型庫（https://fireworks.ai/models）。
> 注意：`reasoning_effort` 僅對推理型模型（如 gpt-oss）有效；非推理模型請設為 `null`。

### 5.2 換成其他供應商（只改 endpoint）
agent 使用通用的 **chat.completions + function calling**，因此換供應商只需改
`base_url` 與 `api_key_env`：

```yaml
# 範例：改用 OpenAI
agent:
  model: gpt-4.1
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY      # 記得在 .env 加入 OPENAI_API_KEY
```

```yaml
# 範例：改用本機 / HPC 上自架的 OpenAI 相容服務（如 vLLM）
agent:
  model: openai/gpt-oss-120b
  base_url: http://localhost:8001/v1
  api_key_env: FIREWORKS_API_KEY   # 自架服務通常不驗證，可隨意填一個存在的變數
```

judge 同理（改 `judge.base_url` / `judge.model` / `judge.api_key_env`）。

> **自評偏誤提醒**：預設 agent 與 judge 都是 gpt-oss-120b。要更可信的數字，建議
> judge 改用不同（或更強）的模型。

---

## 6. 切換資料集／資料來源

「資料」可在三個層次切換：

### 6.1 切換要評估的題目子集（同一個 BrowseComp-Plus）
```yaml
dataset:
  query_ids: ["769", "770", "771"]   # 指定題目
  # 或
  sample: 100                         # 隨機抽 100 題
  seed: 42
```

### 6.2 切換「資料來源」＝檢索設定（這正是論文要隔離的變因）
```yaml
data_source:
  name: qwen3-0.6b-snippet512
  backend: faiss                                  # faiss | reasonir（bm25 需 Java，預設不啟用）
  index_path: "indexes/qwen3-embedding-0.6b/corpus.shard*.pkl"
  model_name: Qwen/Qwen3-Embedding-0.6B
  k: 5                                            # 每次搜尋回傳幾篇
  snippet_max_tokens: 512                         # 512 = 摘要模式；-1 = 全文模式
  get_document: false                             # true 時提供「讀全文」工具
```
- **本地研究資料（摘要）**：`snippet_max_tokens: 512`（回傳截斷預覽）。
- **文獻全文**：`snippet_max_tokens: -1` 且/或 `get_document: true`（讓 agent 讀整篇）。
- **換語料／換檢索器**：改 `index_path` + `model_name`（例如換成 `qwen3-embedding-8b`），
  或實作自訂檢索器（見下）。

### 6.3 換成「完全不同的資料集 / 自己的語料」
本 pipeline 以三個格式契約為邊界，只要提供下列檔案即可評估任意資料集：
1. **Ground truth**（JSONL，每行）：`{"query_id": "...", "query": "...", "answer": "..."}`
   → 設定 `dataset.ground_truth` 指向它。
2. **Qrels**（TREC 格式 `qid Q0 docid 1`）：作為 evidence 召回的標準答案
   → 設定 `dataset.qrel_evidence`。
3. **可檢索的語料 + 索引**：兩種做法
   - 用本 repo 的稠密檢索：依 `scripts_build_index/qwen3-embed.md` 用 Tevatron 對你的語料
     編碼成 `corpus.shard*.pkl`，並把語料放上 HuggingFace（或改用自訂檢索器）。
   - **自訂檢索器 / 網路來源**：實作一個符合端點契約的 HTTP 服務
     （`GET /health`、`POST /search {query,k} → {results:[{docid,score,title,snippet}]}`、
     `POST /get_document {docid}`），然後在 config 設 `data_source.endpoint: http://host:port`。

---

## 7. 切換 Prompt

```yaml
agent:
  query_template: QUERY_TEMPLATE_NO_GET_DOCUMENT   # 內建模板（見 search_agent/prompts.py）
  system_prompt: |                                 # 額外的系統提示（選用）
    You are a meticulous research assistant...
```
內建模板：
- `QUERY_TEMPLATE`：含「讀全文」與引用 [docid] 的完整版。
- `QUERY_TEMPLATE_NO_GET_DOCUMENT`：只搜尋、要求引用（預設）。
- `QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION`：只搜尋、不要求引用。
- `null`：直接用原始問題。
- 也可直接填入**自訂字串模板**（含 `{question}` 佔位符）。

---

## 8. 評估你自己的 Agent

有三種接入方式（推薦第一種，與語言/框架無關）：

### 8.1 Run-file 契約（外部 agent）
讓你的 agent 自行輸出符合契約的 run JSON 到某資料夾，pipeline 只負責評分：
```yaml
experiment_name: my_agent_v3
agent:
  type: external
  external_runs_dir: runs/my_agent_v3   # 你的 run_*.json 放這裡
```
run JSON 必要欄位（完整見 `docs/fireworks_eval.md`）：
```json
{
  "query_id": "769",
  "status": "completed",
  "retrieved_docids": ["18639", "41759"],
  "tool_call_counts": {"search": 7},
  "result": [{"type": "output_text", "output": "Explanation: ...[18639]\nExact Answer: X\nConfidence: 90%"}],
  "metadata": {"model": "your-model"}
}
```
**硬性規定**：`status` 為 `completed`、`result` 最後一筆必須是 `output_text`、最終答案
建議以 `[docid]` 標註引用。

### 8.2 Python Adapter（行程內整合）
實作 `fireworks_eval/agent_protocol.py` 的 `DeepResearchAgent.run(query_id, question, retrieval)`，
提供工廠函式並於 config 指定：
```yaml
agent:
  type: "python:my_pkg.my_agent:make_agent"   # make_agent(agent_cfg) -> DeepResearchAgent
```

### 8.3 內建參考 agent（預設）
`type: fireworks_reference` — 可攜的 chat.completions + tool-calling agent，開箱即用，
也可當你開發的基準線。

---

## 9. 回歸對照

每次優化後，比較兩個實驗，看是**提升**還是有**副作用**：
```bash
python -m fireworks_eval.compare \
  --baseline evals/my_run \
  --candidate evals/my_run_v2 \
  --out reports/cmp_v1_v2
```
輸出 `comparison.md` / `comparison.json`：指標 Δ、逐題 ✗→✓（improvements）與
✓→✗（regressions）、分領域 Δ，以及一句 verdict（`improved` / `regressed` /
`mixed (...)` / `no-change`）。**regressions 就是要盯的副作用。**

---

## 10. 分領域評估

資料集本身沒有 domain 標籤，可用 LLM 自動分類（會快取，最多分類一次）：
```yaml
tagging:
  enabled: true
  taxonomy: configs/taxonomy/default.json   # 可自行增刪領域
```
或自帶標籤：`dataset.tags_file: path/to/{qid: domain}.json`。
之後報表與 summary 會多出 `per_domain` 區塊，讓你看出 agent 在哪些領域強／弱。

---

## 11. 輸出與指標說明

| 產物 | 路徑 |
|---|---|
| 每題 run（agent 軌跡） | `runs/<name>/run_<ts>_<qid>.json` |
| 每題 eval（judge 結果） | `evals/<name>/run_<ts>_<qid>_eval.json` |
| 彙整指標 | `evals/<name>/evaluation_summary.json` |
| 明細 CSV | `evals/<name>/detailed_judge_results.csv` |
| 報表 | `reports/<name>/report.md` |

指標定義（對應論文 §4.4）：
- **Accuracy**：LLM-as-judge 判定語意正確的比例。
- **Recall**：每題「檢索到的 evidence ∩ 標準 evidence / 標準 evidence」之平均。
- **Avg search calls**：平均每題搜尋次數。
- **Calibration Error**：信心與正確率的校準誤差（需 ≥100 題才計算，否則為 `null`）。
- **Citation P/R**：引用的 docid 與 evidence 標註的精確率／召回率。

完整的五大 I/O 契約格式（config / run / eval / summary / comparison）見
**`docs/fireworks_eval.md`**。

---

## 12. E2E Smoke Test

```bash
python -m fireworks_eval.smoke_test                 # 預設題目 769,770,771
python -m fireworks_eval.smoke_test --sample 3 --seed 7
python -m fireworks_eval.smoke_test --ids 769,800,1200 --reasoning-effort high
```
會跑完整 pipeline 並驗證：每題都有合法 run 檔、summary 指標在合理範圍、報表存在、
完成題目無 judge 解析錯誤；並附合理性檢查（有作答、≥1 次搜尋、≥1 篇檢索）。
通過回傳 exit code 0、否則 1（適合接 CI）。

---

## 13. 自動化測試

```bash
uv pip install pytest          # 首次
python -m pytest tests/ -q     # 離線單元測試（不需網路/GPU）
```
涵蓋評分、run-file 契約、config、judge 彙整、回歸對照。整合層（agent/檢索）由 E2E
smoke 實機驗證。

---

## 14. 遷移到 HPC Cluster

核心觀念：**在有網路的登入節點（login node）把所有東西準備好並快取，計算節點
（compute node）以離線模式執行。** 以下分「準備清單 → 預先下載 → 離線環境變數 →
LLM 選項 → SLURM 範例 → 預檢清單」。

### 14.1 需要預先準備到 HPC 的東西
| # | 項目 | 說明 |
|---|---|---|
| 1 | 程式碼 | 本 repo（`git clone` 或打包上傳） |
| 2 | Python 環境 | uv + Python 3.10；用 `uv sync` 重建 `.venv` |
| 3 | 解密後資料集 | `data/browsecomp_plus_decrypted.jsonl`（產生一次後即為靜態檔） |
| 4 | Qrels | `topics-qrels/qrel_evidence.txt`、`qrel_golds.txt`（已隨 repo） |
| 5 | 檢索索引 | `indexes/qwen3-embedding-0.6b/`（靜態檔） |
| 6 | 文件語料 | HF `Tevatron/browsecomp-plus-corpus`（執行期載入 → 須預先快取） |
| 7 | Embedding 模型 | HF `Qwen/Qwen3-Embedding-0.6B`（檢索服務載入 → 須預先快取） |
| 8 | Snippet 斷詞器 | HF `Qwen/Qwen3-0.6B`（截斷 snippet 用 → 須預先快取） |
| 9 | LLM 取得方式 | 走 Fireworks（需對外網路）**或**自架本地 LLM（見 §14.4） |

> 第 3、5 項是**純檔案**，準備一次即可隨意搬移；第 6、7、8 項是 HuggingFace 快取，
> 離線時必須事先下載到固定的 `HF_HOME`。

### 14.2 在登入節點預先下載（一次性）
建議把 HuggingFace 快取放在**共享且持久**的路徑（讓所有計算節點都讀得到）：

```bash
# 1) 固定 HF 快取位置（放進你的 ~/.bashrc 或 job 腳本）
export HF_HOME=/work/$USER/hf_cache          # 換成你的共享儲存路徑
mkdir -p "$HF_HOME"

# 2) 重建 Python 環境
cd /path/to/BrowseComp-Plus-Local
uv sync && source .venv/bin/activate

# 3) 解密資料集（產生靜態 JSONL；需 HF 登入或 HF_TOKEN）
huggingface-cli login          # 或 export HF_TOKEN=hf_xxx
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv

# 4) 下載檢索索引（靜態檔）
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset --include="qwen3-embedding-0.6b/*" --local-dir ./indexes

# 5) 預先快取「執行期會用到」的模型與語料到 HF_HOME
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
huggingface-cli download Qwen/Qwen3-0.6B
huggingface-cli download Tevatron/browsecomp-plus-corpus --repo-type dataset

# 6) 關鍵：實際 build 出「datasets 的 arrow 快取」（執行期用 load_dataset 讀取）
python -c "from datasets import load_dataset; load_dataset('Tevatron/browsecomp-plus-corpus', split='train')"
```

> 第 6 步很重要：`huggingface-cli download ... --repo-type dataset` 只抓原始檔，
> 執行期的 `load_dataset()` 還會建立 arrow 快取。先在有網路時跑一次 `load_dataset`，
> 之後離線才能直接命中快取。

### 14.3 計算節點的離線環境變數
在 compute node（或 SLURM 腳本）設定：

```bash
export HF_HOME=/work/$USER/hf_cache          # 與下載時相同
export HF_HUB_OFFLINE=1                       # 禁止連 HF Hub
export TRANSFORMERS_OFFLINE=1                  # transformers 離線
export HF_DATASETS_OFFLINE=1                   # datasets 離線
# 下列兩個是本 pipeline 在 macOS/CPU 需要的；HPC GPU 環境設了也無害
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
```
> `retrieval_server.py` 已內建 `KMP_DUPLICATE_LIB_OK`、`OMP_NUM_THREADS=1` 的預設值；
> 在 **GPU** 節點可放寬：見 §14.6。

### 14.4 LLM 取得方式（兩選一）
- **(A) HPC 可對外連 Fireworks**：最簡單。確保計算節點能連 `api.fireworks.ai`，
  並設好 `FIREWORKS_API_KEY`，config 不用改。
- **(B) 完全離線**：在 GPU 節點自架一個 **OpenAI 相容**的推理服務（例如 vLLM），
  把 agent/judge 的 `base_url` 指過去：
  ```bash
  # 先在登入節點下載要用的 LLM（例如）
  huggingface-cli download openai/gpt-oss-120b
  # 在 GPU 計算節點啟動 OpenAI 相容服務
  vllm serve openai/gpt-oss-120b --port 8001
  ```
  ```yaml
  agent:  { model: openai/gpt-oss-120b, base_url: http://127.0.0.1:8001/v1, api_key_env: FIREWORKS_API_KEY }
  judge:  { model: openai/gpt-oss-120b, base_url: http://127.0.0.1:8001/v1, api_key_env: FIREWORKS_API_KEY }
  ```
  （`api_key_env` 仍需指向一個存在的環境變數，vLLM 不會驗證其值。）

### 14.5 SLURM 範例（單節點 GPU；LLM 走 Fireworks）
```bash
#!/bin/bash
#SBATCH --job-name=bcp-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- 環境 ---
export HF_HOME=/work/$USER/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
set -a; source .env; set +a          # 載入 FIREWORKS_API_KEY

# --- 執行（pipeline 會自動在本節點啟動檢索服務並於結束時關閉） ---
python -m fireworks_eval.run_experiment \
  --config configs/experiment.example.yaml \
  --experiment-name hpc_run_$SLURM_JOB_ID \
  --sample 830                          # 全量；或先用小樣本試跑
```
提交：`sbatch scripts/run_eval.sbatch`。

> 注意：pipeline 預設以 `endpoint: auto` 在**同一節點**用 subprocess 啟動檢索服務
> （localhost:8123），無需另開服務。若你想讓檢索服務與 agent 分屬不同節點，
> 自行啟動 `python -m fireworks_eval.retrieval_server ...` 後，把 `data_source.endpoint`
> 設成該服務的 URL（並注意 HPC 防火牆／port）。

### 14.6 GPU 加速注意事項
- 有 GPU 時把 `data_source.torch_dtype` 設為 `float16`（或留 `auto`，有 CUDA 會自動用
  float16），檢索編碼會快非常多（CPU 上首次冷啟動約 20 秒、之後 <1 秒；GPU 更快）。
- 記憶體足夠時可改用更強的索引（`qwen3-embedding-8b`），通常顯著提升 Recall 與 Accuracy。
- GPU 節點通常不會有 faiss/torch 的 OpenMP 衝突；`OMP_NUM_THREADS=1` 可移除以提速
  （若出現 segfault 再加回）。

### 14.7 離線預檢清單
執行前確認（compute node 上）：
- [ ] `data/browsecomp_plus_decrypted.jsonl` 存在
- [ ] `indexes/qwen3-embedding-0.6b/corpus.shard*.pkl` 存在（4 個分片）
- [ ] `topics-qrels/qrel_evidence.txt` 存在
- [ ] `HF_HOME` 指向已快取 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-0.6B`、
      `Tevatron/browsecomp-plus-corpus`（含 arrow 快取）的路徑
- [ ] LLM 可用：能連 Fireworks（A）或本地 vLLM 服務已啟動（B）
- [ ] `.env` 內 `FIREWORKS_API_KEY`（或對應金鑰）已設定
- [ ] 先用 `--sample 3` 或 `python -m fireworks_eval.smoke_test` 小樣本試跑成功，再全量

---

## 15. 參數速查表

| Config 欄位 | 用途 | 預設 |
|---|---|---|
| `experiment_name` | 實驗名稱（決定輸出路徑） | （必填） |
| `dataset.query_ids` / `dataset.sample` | 指定題目 / 隨機抽樣 | 全 830 題 |
| `dataset.ground_truth` | ground truth JSONL | `data/browsecomp_plus_decrypted.jsonl` |
| `dataset.qrel_evidence` | evidence 召回標準 | `topics-qrels/qrel_evidence.txt` |
| `data_source.backend` | `faiss` / `reasonir` | `faiss` |
| `data_source.index_path` | FAISS 索引 glob | `indexes/qwen3-embedding-0.6b/corpus.shard*.pkl` |
| `data_source.model_name` | embedding 模型 | `Qwen/Qwen3-Embedding-0.6B` |
| `data_source.k` | 每次搜尋回傳數 | 5 |
| `data_source.snippet_max_tokens` | 摘要長度（-1=全文） | 512 |
| `data_source.get_document` | 提供讀全文工具 | false |
| `data_source.endpoint` | `auto` 或檢索服務 URL | auto |
| `data_source.torch_dtype` | `auto`/`float16`/`float32` | auto |
| `agent.type` | `fireworks_reference`/`external`/`python:...` | fireworks_reference |
| `agent.model` / `agent.base_url` / `agent.api_key_env` | agent LLM 與端點 | gpt-oss-120b / Fireworks |
| `agent.query_template` / `agent.system_prompt` | 提示 | NO_GET_DOCUMENT / 無 |
| `agent.reasoning_effort` | low/medium/high/null | high |
| `agent.max_tokens` / `max_iterations` / `num_threads` | 產生上限 / 迴圈上限 / 併發 | 10000 / 50 / 4 |
| `judge.model` / `judge.base_url` / `judge.api_key_env` | judge LLM 與端點 | gpt-oss-120b / Fireworks |
| `tagging.enabled` / `tagging.taxonomy` | 分領域 | false / default.json |

CLI 覆寫（`run_experiment`）：`--experiment-name`、`--sample`、`--query-ids`、
`--tagging/--no-tagging`、`--force-agent`、`--force-judge`、`--rebuild-gt-cache`。

---

## 16. 常見問題排解

| 症狀 | 解法 |
|---|---|
| `FIREWORKS_API_KEY not set` | 在 `.env` 設定金鑰；HPC 上 `set -a; source .env; set +a` |
| `No module named pip` | 本 venv 由 uv 管理，請用 `uv pip install ...` |
| `You must install Java` | 只影響 BM25；本 pipeline 預設走 dense FAISS，不需 Java |
| 檢索服務 `not healthy within Ns` | 看 `tmp/retrieval_server_<port>.log`；首次載入較久，可調高 `data_source.startup_timeout_s` |
| segfault（exit 139） | faiss/torch OpenMP 衝突；確認 `KMP_DUPLICATE_LIB_OK=TRUE`、`OMP_NUM_THREADS=1`（CPU） |
| `flash_attn ... not installed` | 已內建以 `eager` attention 載入；若自行載模型請設 `attn_implementation="eager"` |
| 離線環境連線失敗（HF） | 確認 `HF_HOME` 已快取所需模型/語料，且設了 `HF_HUB_OFFLINE=1` 等離線旗標 |
| port 被占用 | 改 `data_source.port`；或 pipeline 會自動重用已啟動的健康端點 |
| CPU 很慢 | 正常；開發時用小樣本，正式評估移到 GPU 節點並用 `float16` |

---

更深入的 I/O 契約與設計細節：見 [`docs/fireworks_eval.md`](fireworks_eval.md) 與
[`docs/superpowers/specs/2026-06-14-fireworks-e2e-smoke-eval-design.md`](superpowers/specs/2026-06-14-fireworks-e2e-smoke-eval-design.md)。
