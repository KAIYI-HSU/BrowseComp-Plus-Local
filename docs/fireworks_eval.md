# fireworks_eval — Deep-Research Agent Evaluation Pipeline

A 100%-automated evaluation pipeline built on the **BrowseComp-Plus** benchmark for
measuring a Deep-Research agent's capability and accuracy — and, crucially, for
detecting whether a change (model, prompt, or data source) **improved** results or
introduced **side-effects**.

- **LLM calls** go through an OpenAI-compatible endpoint — **Fireworks** by default
  (`FIREWORKS_API_KEY`). Switch provider/model by changing `base_url`/`model`.
- **Retrieval/embeddings** run **locally** (Qwen3-Embedding-0.6B + FAISS) behind a
  managed local HTTP endpoint.

---

## 1. Quick start

```bash
# 0. Ensure .env has FIREWORKS_API_KEY=... and the Qwen3-0.6B FAISS index exists at
#    indexes/qwen3-embedding-0.6b/  (already present in this repo).

# 1. E2E smoke test — 3 queries through the full pipeline + output validation:
.venv/bin/python -m fireworks_eval.smoke_test

# 2. A full experiment from a config file:
.venv/bin/python -m fireworks_eval.run_experiment --config configs/experiment.example.yaml

# 3. A quick subset / sample:
.venv/bin/python -m fireworks_eval.run_experiment --config configs/smoke.yaml --sample 20 --tagging

# 4. Compare two experiments (improvement vs side-effects):
.venv/bin/python -m fireworks_eval.compare --baseline evals/exp_a --candidate evals/exp_b
```

The first run builds a one-time slim ground-truth cache (`data/ground_truth_slim.jsonl`)
and, when `endpoint: auto`, starts the local retrieval server (first start loads the
embedding model + corpus, ~1–2 min on CPU). Everything is resumable.

---

## 2. Architecture & data flow

```
ExperimentConfig (YAML/JSON)
        │
        ▼
  run_experiment ──► pipeline.run_pipeline
   1. dataset.py      select query subset (ids | sample) from slim ground truth
   2. tagging.py      (optional) classify each query → domain (Fireworks, cached)
   3. retrieval       managed local endpoint: Qwen3-Embedding-0.6B + FAISS   ◄── data source
   4. agent.py        FireworksDeepResearchAgent → runs/<exp>/run_*.json     ◄── agent under test
   5. judge.py        LLM-as-judge (Fireworks) → evals/<exp>/*_eval.json + summary + csv
   6. report.py       reports/<exp>/report.md (overall + per-domain + per-query)
        │
        ▼
  compare.py  (baseline summary) vs (candidate summary) → deltas + flips + verdict
```

Each stage writes files to disk and is **resumable** — a re-run skips queries with
existing run/eval files (use `--force-agent` / `--force-judge` to override).

---

## 3. The three swap dimensions (what this pipeline is for)

| Dimension | Config knobs | Example |
|---|---|---|
| **Model per stage** | `agent.model`, `judge.model` | swap agent to `accounts/fireworks/models/qwen3-...` while keeping the judge fixed |
| **Prompt** | `agent.query_template`, `agent.system_prompt` | `QUERY_TEMPLATE` (with full-doc + citations) vs `..._NO_GET_DOCUMENT` vs a custom literal template |
| **Data source** | `data_source.{backend,index_path,model_name,k,snippet_max_tokens,get_document}` | summary mode (`snippet_max_tokens: 512`) vs full-text (`snippet_max_tokens: -1`, `get_document: true`); or a different corpus/retriever |

To switch **provider** (Fireworks → other), change `base_url` (+ `api_key_env`). The agent
uses the portable `chat.completions` + tool-calling API.

### Data-source semantics (本地研究資料 / 網路文獻摘要 / 文獻全文)
- **Local research data**: the local Qwen3-Embedding FAISS corpus (default).
- **Summaries**: small `snippet_max_tokens` → search returns truncated previews.
- **Full text**: `snippet_max_tokens: -1` and/or `get_document: true` → agent reads full docs.
- **Web sources**: implement a custom retriever behind the same endpoint contract
  (see §6.2) and point `data_source.endpoint` at it.

---

## 4. Input contract — Experiment config

See `configs/experiment.example.yaml` for the fully-commented reference. Minimal config:

```yaml
experiment_name: my_run
dataset:
  query_ids: ["769", "770", "771"]   # or:  sample: 50
agent:
  model: accounts/fireworks/models/gpt-oss-120b
  query_template: QUERY_TEMPLATE_NO_GET_DOCUMENT
judge:
  model: accounts/fireworks/models/gpt-oss-120b
```

Sections: `dataset`, `data_source`, `agent`, `judge`, `tagging`, `output`. Unknown keys
are ignored with a warning. `{experiment_name}` is substituted into output paths.

---

## 5. Output contracts (concrete formats)

### 5.1 Run file — `runs/<exp>/run_<ts>_<qid>.json` (the universal agent boundary)

```json
{
  "schema_version": "1.0",
  "query_id": "769",
  "status": "completed",
  "retrieved_docids": ["18639", "41759"],
  "tool_call_counts": {"search": 7},
  "result": [
    {"type": "tool_call", "tool_name": "search", "arguments": "{\"query\": \"...\"}", "output": "[{...}]"},
    {"type": "output_text", "output": "Explanation: ... [18639]\nExact Answer: X\nConfidence: 90%"}
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
  "metadata": {"model": "accounts/fireworks/models/gpt-oss-120b", "data_source": "...", "experiment": "..."}
}
```

**Hard requirements** (validated; this is what makes any agent pluggable):
- `status == "completed"` for a scorable answer (else treated as failure).
- the **last** `result` item is `{"type": "output_text", "output": <final answer>}`.
- `retrieved_docids`, `tool_call_counts`, `query_id` present; `metadata.model` labels the summary.
- the final answer should cite evidence as `[docid]` (used for citation metrics).

### 5.2 Eval file — `evals/<exp>/run_<ts>_<qid>_eval.json` (per query)

```json
{
  "json_path": "runs/.../run_..._769.json",
  "query_id": "769",
  "question": "...",
  "response": "<agent final answer>",
  "correct_answer": "Queen Arwa University",
  "is_completed": true,
  "judge_prompt": "...",
  "judge_response": "extracted_final_answer: ...\ncorrect: yes\nconfidence: 99%",
  "judge_result": {"extracted_final_answer": "...", "reasoning": "...", "correct": true, "confidence": 99.0, "parse_error": false},
  "tool_call_counts": {"search": 7},
  "citations": {"cited_docids": ["18639"], "metrics": {"num_citations": 1, "num_relevant": 6, "precision": 1.0, "recall": 0.17}},
  "retrieval": {"retrieved_docids": ["..."], "recall": 0.83},
  "model_info": {"judge_model": "accounts/fireworks/models/gpt-oss-120b", "max_output_tokens": 1024},
  "domain": "History"
}
```

### 5.3 Summary — `evals/<exp>/evaluation_summary.json`

Leaderboard-compatible keys plus extensions:

```json
{
  "LLM": "accounts/fireworks/models/gpt-oss-120b",
  "Accuracy (%)": 66.67,
  "Recall (%)": 72.4,
  "avg_tool_stats": {"search": 8.3},
  "Calibration Error (%)": null,
  "Citation": {"coverage_%": 100.0, "avg_citations": 3.2, "precision_%": 61.0, "recall_%": 38.0},
  "num_queries": 3, "num_completed": 3, "num_parse_errors": 0,
  "experiment": "my_run", "data_source": "qwen3-0.6b-snippet512",
  "agent_model": "...", "judge_model": "...",
  "per_domain": {"History": {"n": 1, "accuracy_%": 100.0, "recall_%": 83.3}},
  "per_query_metrics": [{"query_id": "769", "correct": true, "recall": 83.3, "domain": "History"}]
}
```

Also emitted: `detailed_judge_results.csv`, and `reports/<exp>/report.md` + `summary.json`.

### 5.4 Comparison — `reports/comparison/comparison.json` + `.md`

```json
{
  "baseline": "exp_a", "candidate": "exp_b",
  "deltas": {"accuracy_%": 6.67, "recall_%": 2.1, "avg_search_calls": -1.2, "calibration_%": -3.0},
  "improvements": [{"query_id": "769"}],
  "regressions": [{"query_id": "770"}],
  "unchanged": 1, "shared_queries": 3,
  "per_domain_deltas": {"History": {"accuracy_%": 0.0, "recall_%": 5.0, "n": 1}},
  "verdict": "mixed (net gain, with regressions)"
}
```

`verdict` ∈ `improved | regressed | mixed (...) | no-change`. **Regressions** (✓→✗) are the
side-effects to watch after each optimization.

---

## 6. Plugging in YOUR agent

### 6.1 Run-file contract (recommended — language-agnostic)
Have your agent write conforming run files (§5.1) into a directory, then evaluate them:

```yaml
experiment_name: my_agent_v3
agent:
  type: external
  external_runs_dir: runs/my_agent_v3   # your run_*.json live here
```

```bash
python -m fireworks_eval.run_experiment --config my_agent.yaml
```
The pipeline skips agent execution and only judges + reports.

### 6.2 Python adapter (in-process)
Implement `DeepResearchAgent` (`fireworks_eval/agent_protocol.py`):

```python
def run(self, query_id: str, question: str, retrieval) -> RunRecord: ...
```
`retrieval` is a `RetrievalBackend` with `.search(query, k)` and `.get_document(docid)`.
Expose a factory and reference it:

```yaml
agent:
  type: "python:my_pkg.my_agent:make_agent"   # make_agent(agent_cfg) -> DeepResearchAgent
```

### 6.3 Built-in reference agent (default)
`type: fireworks_reference` — a portable chat.completions + tool-calling agent. Use it as
a baseline or a starting point.

### Custom retriever / web source
Either run a service that implements the endpoint contract
(`GET /health`, `POST /search {query,k} -> {results:[{docid,score,title,snippet}]}`,
`POST /get_document {docid}`) and set `data_source.endpoint: http://host:port`, or subclass
the repo's `BaseSearcher` (`searcher/searchers/`) and serve it via
`fireworks_eval.retrieval_server`.

---

## 7. Per-domain (scenario) evaluation

The dataset has no domain labels, so domains are assigned by an LLM and cached:

```yaml
tagging:
  enabled: true
  taxonomy: configs/taxonomy/default.json    # edit the domain list as you like
```
Or supply your own labels: `dataset.tags_file: path/to/{qid: domain}.json`.
The report and summary then break metrics down per domain, so you can see *where* the
agent is strong or weak.

---

## 8. Metric definitions (mapped to the paper, §4.4)

| Metric | Meaning |
|---|---|
| **Accuracy** | fraction of answers judged semantically correct (LLM-as-judge, yes/no) |
| **Recall** | mean over queries of `|retrieved ∩ evidence_qrels| / |evidence_qrels|` |
| **Avg search calls** | mean tool calls per query |
| **Calibration Error** | RMS gap between stated confidence and correctness (needs ≥100 queries; `null` otherwise) |
| **Citation P/R** | cited docids vs evidence qrels |

The grader prompt, parsing, and calibration are faithful replications of
`scripts_evaluation/evaluate_run.py` (we replicate rather than import because that module
hard-imports `vllm`). Results are therefore directly comparable to the official evaluator,
with the judge running on Fireworks instead of local vLLM.

> **Judge bias note**: by default the agent and judge use the same model
> (`gpt-oss-120b`). For higher-confidence numbers, set a different `judge.model`.

---

## 9. E2E smoke test

`python -m fireworks_eval.smoke_test` samples 3 queries (default `769,770,771`), runs the
full pipeline, and validates the outputs:

- **Critical** (gate PASS/FAIL): every selected query has a schema-valid run file; the
  summary exists with `Accuracy`/`Recall` in range; `per_query_metrics` count matches;
  the report exists; no judge parse errors on completed runs.
- **Advisory** (warn): completed runs have a non-empty answer, ≥1 search call, ≥1 retrieved
  doc; at least one run completed.

Exit code `0` on PASS, `1` on FAIL. Options: `--sample N --seed S`, `--ids a,b,c`,
`--max-tokens`, `--threads`, `--tagging`.

Offline tests (no network/GPU) cover scoring, the run-file contract, config, and the judge
aggregation on synthetic fixtures:

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `FIREWORKS_API_KEY not set` | add it to `.env` |
| retrieval server "not healthy within Ns" | check `tmp/retrieval_server_<port>.log`; first start is slow (model+corpus load) — raise `data_source.startup_timeout_s` |
| `You must install Java` | only affects BM25; the dense (FAISS) path is used by default and needs no Java |
| port already in use | change `data_source.port`, or it will reuse an already-healthy endpoint |
| slow on Mac CPU | expected; keep subsets small for dev loops; the endpoint loads the model once and serves all queries |
