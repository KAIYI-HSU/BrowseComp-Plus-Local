# BrowseComp-Plus 評估流程

本文件說明如何在本機重複執行 BrowseComp-Plus 的 Fireworks-first
評估流程。目標是讓開發者可以固定資料集與檢索器，快速比較不同
LLM、prompt、retriever 或 knowledge source 對 Deep Research agent 的影響。

本流程使用：

- 固定的 BrowseComp-Plus corpus，不使用 live web search
- 830 題人工驗證的 decrypted BrowseComp-Plus QA
- 本機 search tool，每次預設回傳 top-k 文件，預設 `k=5`
- 每個 retrieved snippet 預設最多 512 tokens
- Fireworks/OpenAI-compatible Chat Completions 執行 agent 與 judge
- end-to-end answer accuracy
- agent 全軌跡 retrieved documents 的 retrieval recall
- 平均 search calls、calibration error、citation precision/recall

主要檔案：

- `search_agent/fireworks_client.py`：透過 Fireworks Chat Completions 執行
  tool-using agent，並輸出 BrowseComp-Plus run JSON。
- `searcher/searchers/openai_embedding_faiss_searcher.py`：透過本機
  OpenAI-compatible embeddings endpoint 產生 query embedding，並查詢本機
  prebuilt FAISS corpus index。
- `scripts_evaluation/evaluate_with_fireworks.py`：用 Fireworks/OpenAI-compatible
  judge 評估既有 run directory。
- `scripts_evaluation/pipeline.py`：選用的 scenario runner，可批次跑多組
  experiment 並產生 comparison summary。
- `configs/fireworks_browsecomp_plus.example.json`：BM25 與
  Qwen3-Embedding-0.6B 實驗 manifest 範例。

## 基本設定

```bash
uv sync
source .venv/bin/activate

# .env 需要包含：
# FIREWORKS_API_KEY=...
```

如果尚未產生 decrypted dataset 與 query TSV：

```bash
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

若只需要 Qwen3-Embedding-0.6B index，可只下載該目錄：

```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset \
  --include="qwen3-embedding-0.6b/*" \
  --local-dir ./indexes
```

也可以下載全部 index：

```bash
bash scripts_build_index/download_indexes.sh
```

預設模型與 endpoint：

- agent/judge model：`accounts/fireworks/models/gpt-oss-120b`
- Fireworks base URL：`https://api.fireworks.ai/inference/v1`
- Fireworks API key env var：`FIREWORKS_API_KEY`
- embedding model：`Qwen/Qwen3-Embedding-0.6B`
- embedding API base URL：`http://localhost:8000/v1`
- embedding API key：`EMPTY`

`gpt-oss-120b` 是推理模型。Smoke test 與正式評估建議使用較大的 token
預算，例如 agent `--max-tokens 10000`，judge `--max-output-tokens 10000`，
避免模型只輸出 reasoning 或被截斷，造成沒有可評估的 final answer。

## 啟動本機 Qwen3-Embedding Endpoint

Fireworks 用於 LLM 與 judge；Qwen3-Embedding-0.6B 由本機
OpenAI-compatible endpoint 提供。vLLM 範例：

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

另一個終端機檢查 endpoint 是否可用：

```bash
python - <<'PY'
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-0.6B",
    input="test query",
)
print(len(response.data[0].embedding))
PY
```

成功時應輸出 embedding 維度；Qwen3-Embedding-0.6B 通常是 `1024`。

## 必要輸入格式

### Ground Truth JSONL

預設路徑：`data/browsecomp_plus_decrypted.jsonl`

每一行是一個 JSON object：

```json
{
  "query_id": "q1",
  "query": "Question text",
  "answer": "Precise ground-truth answer"
}
```

### Query TSV

預設路徑：`topics-qrels/queries.tsv`

每一列：

```text
query_id<TAB>query text
```

這是 `search_agent/fireworks_client.py` batch mode 的輸入。

### Evidence Qrels

預設路徑：`topics-qrels/qrel_evidence.txt`

TREC qrel 格式：

```text
query_id 0 docid relevance
```

Evaluator 會用它計算 retrieval recall 與 citation metrics。

### Agent Run JSON

每題一個 JSON 檔。Evaluator 會忽略額外欄位，但下列欄位是必要的：

```json
{
  "query_id": "q1",
  "tool_call_counts": {
    "search": 3
  },
  "status": "completed",
  "retrieved_docids": ["123", "456"],
  "result": [
    {
      "type": "output_text",
      "output": "Explanation: ... [123].\nExact Answer: ...\nConfidence: 80%"
    }
  ]
}
```

規則：

- `query_id` 必須是字串，且必須存在於 ground truth JSONL。
- `status` 只有在 agent 真的輸出 final answer 時才應為 `"completed"`。
- 非 `"completed"` 的 records 會被視為 incorrect，但仍會納入 retrieval 與
  tool-call metrics。
- `retrieved_docids` 應該是整段 agent trajectory 中所有 retrieved documents
  的 union，不只包含 citation 中的文件。
- final answer 應該是最後一個 `type == "output_text"` 的 result item。
- citation 建議使用 BrowseComp-Plus numeric docid，例如 `[123]` 或
  `[123, 456]`，以便計算 citation metrics。

## E2E Smoke Test：3 題人工逐步檢查

這個 smoke test 使用真實 BrowseComp-Plus QA，不產生合成題。它適合在修改
agent、retriever、prompt 或 evaluator 後快速檢查完整鏈路。

### 1. 檢查 Fireworks API key

```bash
test -f .env && grep -q '^FIREWORKS_API_KEY=' .env \
  && echo "FIREWORKS_API_KEY exists" \
  || echo "Missing FIREWORKS_API_KEY"
```

### 2. 準備 decrypted dataset 與 query TSV

```bash
test -f data/browsecomp_plus_decrypted.jsonl \
  && test -f topics-qrels/queries.tsv \
  || python scripts_build_index/decrypt_dataset.py \
      --output data/browsecomp_plus_decrypted.jsonl \
      --generate-tsv topics-qrels/queries.tsv
```

檢查筆數：

```bash
wc -l data/browsecomp_plus_decrypted.jsonl topics-qrels/queries.tsv
```

### 3. 準備 Qwen3-Embedding-0.6B FAISS index

```bash
find indexes/qwen3-embedding-0.6b -maxdepth 1 \
  -type f -name 'corpus.shard*.pkl' | wc -l
```

如果輸出不是 `4`，下載 index：

```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset \
  --include="qwen3-embedding-0.6b/*" \
  --local-dir ./indexes
```

### 4. 啟動或檢查本機 embedding endpoint

檢查是否已有服務：

```bash
curl -sS -m 5 http://localhost:8000/v1/models | python -m json.tool
```

若尚未啟動：

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

Embedding smoke test：

```bash
python - <<'PY'
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-0.6B",
    input="smoke test embedding",
)
print({"vectors": len(response.data), "dimension": len(response.data[0].embedding)})
PY
```

### 5. 建立 3 題 smoke TSV

```bash
head -n 3 topics-qrels/queries.tsv > /tmp/browsecomp_plus_e2e_3q.tsv
wc -l /tmp/browsecomp_plus_e2e_3q.tsv
sed -n '1,3p' /tmp/browsecomp_plus_e2e_3q.tsv
```

### 6. 跑 Fireworks agent + local embedding FAISS retrieval

建議使用新的 output directory，避免舊檔被 batch runner 略過：

```bash
RUN_DIR=runs/e2e_smoke/gpt-oss-120b_qwen3-0p6b_3q_tokens10000
mkdir -p "$RUN_DIR"

python search_agent/fireworks_client.py \
  --query /tmp/browsecomp_plus_e2e_3q.tsv \
  --model accounts/fireworks/models/gpt-oss-120b \
  --output-dir "$RUN_DIR" \
  --searcher-type openai-embedding-faiss \
  --index-path 'indexes/qwen3-embedding-0.6b/corpus.shard*.pkl' \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-base-url http://localhost:8000/v1 \
  --embedding-api-key EMPTY \
  --normalize \
  --reasoning-effort high \
  --max-iterations 20 \
  --max-tokens 10000 \
  --num-threads 1
```

檢查 run JSON：

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/e2e_smoke/gpt-oss-120b_qwen3-0p6b_3q_tokens10000")
paths = sorted(run_dir.glob("*.json"))
print("run_json_count", len(paths))
for path in paths:
    record = json.loads(path.read_text())
    has_output = any(
        item.get("type") == "output_text" and item.get("output")
        for item in record.get("result", [])
    )
    print(path.name, {
        "query_id": record.get("query_id"),
        "status": record.get("status"),
        "tool_call_counts": record.get("tool_call_counts"),
        "retrieved_docids": len(record.get("retrieved_docids") or []),
        "has_output_text": has_output,
    })
PY
```

### 7. 跑 Fireworks judge 評估

```bash
python scripts_evaluation/evaluate_with_fireworks.py \
  --input_dir runs/e2e_smoke/gpt-oss-120b_qwen3-0p6b_3q_tokens10000 \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --qrel_evidence topics-qrels/qrel_evidence.txt \
  --eval_dir evals/e2e_smoke_tokens10000 \
  --model accounts/fireworks/models/gpt-oss-120b \
  --reasoning-effort low \
  --max-output-tokens 10000 \
  --force
```

檢查 summary：

```bash
python - <<'PY'
import json
from pathlib import Path

summary_path = Path(
    "evals/e2e_smoke_tokens10000/e2e_smoke/"
    "gpt-oss-120b_qwen3-0p6b_3q_tokens10000/evaluation_summary.json"
)
summary = json.loads(summary_path.read_text())
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

### 8. Smoke test 驗收條件

- 本機 embedding endpoint 可回傳 vector。
- Agent 產生 exactly 3 個 run JSON。
- 每個 run JSON 有合法 `query_id`。
- 至少一個 run 有非空 `retrieved_docids`。
- Evaluator 產生 `evaluation_summary.json`。
- Summary 有 numeric `Accuracy (%)` 與 `Recall (%)`。
- Summary 有 `avg_tool_stats`、`Completed`、`Incomplete`、`Parse Errors`。
- 若有失敗，記錄失敗 command、stderr 摘要與對應 artifact path。

## 跑完整 Agent Scenario

BM25 範例：

```bash
python search_agent/fireworks_client.py \
  --query topics-qrels/queries.tsv \
  --model accounts/fireworks/models/gpt-oss-120b \
  --output-dir runs/fireworks/gpt-oss-120b/bm25_high \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --reasoning-effort high \
  --max-tokens 10000 \
  --num-threads 4
```

Qwen3-Embedding-0.6B + local OpenAI-compatible embeddings endpoint：

```bash
python search_agent/fireworks_client.py \
  --query topics-qrels/queries.tsv \
  --model accounts/fireworks/models/gpt-oss-120b \
  --output-dir runs/fireworks/gpt-oss-120b/qwen3_0p6b_high \
  --searcher-type openai-embedding-faiss \
  --index-path 'indexes/qwen3-embedding-0.6b/corpus.shard*.pkl' \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-base-url http://localhost:8000/v1 \
  --embedding-api-key EMPTY \
  --normalize \
  --reasoning-effort high \
  --max-tokens 10000 \
  --num-threads 4
```

Prompt variants：

```bash
--query-template QUERY_TEMPLATE
--query-template QUERY_TEMPLATE_NO_GET_DOCUMENT
--query-template QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION
```

每次改模型、prompt、retriever、資料來源或 token budget，都應使用不同
`--output-dir`。該 directory name 會成為穩定的 evaluation unit。

## 評估既有 Runs

```bash
python scripts_evaluation/evaluate_with_fireworks.py \
  --input_dir runs/fireworks/gpt-oss-120b/qwen3_0p6b_high \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --qrel_evidence topics-qrels/qrel_evidence.txt \
  --eval_dir evals/fireworks \
  --model accounts/fireworks/models/gpt-oss-120b \
  --reasoning-effort low \
  --max-output-tokens 10000 \
  --force
```

若要做 leaderboard-style reproduction，仍可使用 upstream local Qwen3-32B judge：

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/fireworks/gpt-oss-120b/qwen3_0p6b_high \
  --tensor_parallel_size 1
```

Fireworks judge 主要用於本機自動化開發迴圈，分數可能與 paper 的主要
Qwen3-32B judge 不完全一致。

## Pipeline Manifest

執行 example manifest 中的所有 scenarios：

```bash
python scripts_evaluation/pipeline.py \
  --manifest configs/fireworks_browsecomp_plus.example.json \
  --force-eval
```

只評估既有 run directories，不重跑 agents：

```bash
python scripts_evaluation/pipeline.py \
  --manifest configs/fireworks_browsecomp_plus.example.json \
  --skip-agent-runs \
  --force-eval
```

只跑單一 scenario：

```bash
python scripts_evaluation/pipeline.py \
  --manifest configs/fireworks_browsecomp_plus.example.json \
  --only gpt-oss-120b_qwen3-0p6b_local_high
```

Manifest 範例：

```json
{
  "ground_truth": "data/browsecomp_plus_decrypted.jsonl",
  "qrel_evidence": "topics-qrels/qrel_evidence.txt",
  "eval_dir": "evals/fireworks",
  "comparison_path": "evals/fireworks/comparison_summary.json",
  "judge": {
    "api_key_env": "FIREWORKS_API_KEY",
    "base_url": "https://api.fireworks.ai/inference/v1",
    "model": "accounts/fireworks/models/gpt-oss-120b",
    "max_output_tokens": 10000,
    "temperature": 0.0,
    "reasoning_effort": "low"
  },
  "scenarios": [
    {
      "name": "scenario_name",
      "run_dir": "runs/scenario_name",
      "agent_command": ["python", "..."]
    }
  ]
}
```

Scenario-level `ground_truth`、`qrel_evidence`、`eval_dir` 與 `judge` 會覆蓋
manifest-level defaults。

## 輸出格式

### Per-Query Eval JSON

路徑：

```text
evals/.../<run_file_stem>_eval.json
```

格式：

```json
{
  "json_path": "runs/.../run.json",
  "query_id": "q1",
  "question": "Question text",
  "response": "Agent final response",
  "correct_answer": "Ground truth",
  "is_completed": true,
  "judge_prompt": "...",
  "judge_response": "...",
  "judge_result": {
    "extracted_final_answer": "...",
    "reasoning": "...",
    "correct": true,
    "confidence": 80.0,
    "parse_error": false
  },
  "tool_call_counts": {"search": 3},
  "citations": {
    "cited_docids": ["123"],
    "metrics": {
      "num_citations": 1,
      "num_relevant": 6,
      "precision": 1.0,
      "recall": 0.1666666667
    }
  },
  "retrieval": {
    "retrieved_docids": ["123", "456"],
    "recall": 0.3333333333
  },
  "model_info": {
    "judge_model": "accounts/fireworks/models/gpt-oss-120b",
    "max_output_tokens": 10000
  }
}
```

### Evaluation Summary

路徑：

```text
evals/.../evaluation_summary.json
```

格式：

```json
{
  "LLM": "accounts/fireworks/models/gpt-oss-120b",
  "Accuracy (%)": 42.0,
  "Recall (%)": 50.0,
  "avg_tool_stats": {"search": 8.5},
  "Completed": 3,
  "Incomplete": 0,
  "Parse Errors": 0,
  "Calibration Error (%)": 30.0,
  "Retriever": "change me when submitting",
  "Link": "change me when submitting",
  "Evaluation Date": "2026-06-14",
  "per_query_metrics": [
    {"query_id": "q1", "correct": true, "recall": 50.0}
  ]
}
```

### Detailed CSV

路徑：

```text
evals/.../detailed_judge_results.csv
```

欄位包含：

- `query_id`
- `predicted_answer`
- `correct_answer`
- `judge_correct`
- `confidence`
- `is_completed`
- `parse_error`
- `json_path`
- `retrieval_recall`
- `num_citations`
- `precision_positives`
- `recall_positives`

### Comparison Summary

路徑：

```text
evals/fireworks/comparison_summary.json
```

第一個 scenario 是 baseline，後續 scenarios 會包含 deltas：

```json
{
  "baseline": "baseline_scenario",
  "scenarios": [
    {
      "name": "candidate",
      "run_dir": "runs/candidate",
      "summary_path": "evals/candidate/evaluation_summary.json",
      "accuracy": 45.5,
      "recall": 55.0,
      "avg_search_calls": 8.5,
      "calibration_error": 18.0,
      "delta_accuracy": 5.5,
      "delta_recall": 5.0,
      "delta_search_calls": -1.5,
      "delta_calibration_error": -2.0
    }
  ]
}
```

## 比較不同資料來源

若要比較 local research data、web literature summaries 與 full-text papers，
請保持 agent output contract 不變，只替換 search tool 背後的 retriever 或
searcher。

建議模式：

1. 實作或設定一個 searcher，回傳：

   ```json
   {"docid": "stable-id", "score": 0.0, "text": "retrieved text"}
   ```

2. 每種資料來源使用不同 scenario 與 run directory：

   ```text
   runs/my_agent/local_research_notes
   runs/my_agent/web_literature_summaries
   runs/my_agent/full_text_papers
   ```

3. 保持相同 query set、judge model 與 evaluation command。

這樣可以隔離資料來源的影響，同時維持可比較的輸出 metrics。

## 開發檢查

離線單元測試：

```bash
python -m unittest discover -s tests
```

每次 optimization 建議 gate：

1. 跑 unit tests。
2. 跑低成本 E2E smoke test。
3. 用 `evaluate_with_fireworks.py` 評估 subset。
4. 若要合併較大的 prompt/model/retriever 改動，再跑完整 manifest。
