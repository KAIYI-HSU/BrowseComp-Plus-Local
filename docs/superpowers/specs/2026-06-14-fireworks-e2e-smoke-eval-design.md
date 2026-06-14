# Fireworks E2E Smoke Eval — Design Spec

**Date:** 2026-06-14
**Branch:** `claude/fireworks-e2e-smoke-eval`
**Status:** Approved (4 architecture decisions confirmed by user)

## 1. Goal

Build a **usable, 100%-automated evaluation pipeline** on top of the BrowseComp-Plus
benchmark to evaluate the user's own Deep-Research agent across scenarios/domains, with
the ability to measure the effect of changing:

1. **Models per stage** — the agent LLM and the judge LLM (independently).
2. **Prompts** — query template / system prompt.
3. **Data source** — retrieval backend / corpus / snippet-vs-full-text.

Every optimization must be checkable for **improvement or side-effects** (regression
comparison). Deliver an **E2E smoke test** (sample 3 questions → full pipeline → validate
output format + sanity) with documentation and examples.

All LLM calls route through **Fireworks** (`FIREWORKS_API_KEY`, OpenAI-compatible),
swappable to any provider by changing `base_url`. Embeddings/retrieval run **locally**
(Qwen3-Embedding-0.6B + FAISS) as a managed local HTTP endpoint.

## 2. Confirmed Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Agent integration | **Run-file contract** (universal boundary) + Python adapter protocol + built-in Fireworks reference agent |
| 2 | Retrieval deployment | **Auto-managed local HTTP endpoint** (Qwen3-Embedding-0.6B + FAISS) |
| 3 | Per-domain eval | **Auto LLM tagging** (Fireworks) + user-supplied tags; report groups by domain |
| 4 | Judge model | **gpt-oss-120b** on Fireworks (configurable; self-judging bias noted) |

## 3. What already exists (reused, not rebuilt)

- **Dataset**: `data/browsecomp_plus_decrypted.jsonl` (830 queries: `query_id, query,
  answer, gold_docs, evidence_docs, negative_docs`). No domain labels.
- **Corpus**: HF `Tevatron/browsecomp-plus-corpus` (~100K docs), cached.
- **Retriever engine**: `searcher/searchers/faiss_searcher.py` (`FaissSearcher`,
  Qwen3-Embedding). FAISS 0.6B index downloaded at `indexes/qwen3-embedding-0.6b/`.
- **Metrics definition** (paper §4.4): Accuracy (LLM-as-judge), Recall (evidence-doc
  recall over agent's retrieved set), Search Calls, Calibration Error, Citation P/R.
- **Judge logic**: `scripts_evaluation/evaluate_run.py` (grader prompt, parsing,
  calibration, citation metrics) — replicated (cannot import: it hard-imports `vllm`).

## 4. Gaps this pipeline fills

1. **Fireworks reference agent** (`oss_client.py` hardcodes `api_key="EMPTY"` + Responses
   API → not portable). New agent uses **chat.completions** (provider-portable) + tool
   calling + `FIREWORKS_API_KEY`.
2. **Fireworks judge** (`evaluate_run.py` needs local GPU/vLLM, unavailable on Mac).
3. **Config-driven orchestration** (one command, fully automated).
4. **Regression comparison** (baseline vs candidate: deltas + per-query flips).
5. **Per-domain slicing** (auto tagging + custom tags).
6. **E2E smoke test** + I/O format documentation.

## 5. Architecture

```
                ExperimentConfig (YAML/JSON)  ── the single input ──┐
                                                                    v
 run_experiment.py ─► pipeline.py ──────────────────────────────────────────────
   1. dataset.py     select query subset (ids | sample) from slim ground truth
   2. tagging.py     (optional) classify queries → domain tags (Fireworks, cached)
   3. retrieval      ManagedRetrievalServer: spawn retrieval_server.py (FAISS +
      _client.py     Qwen3-Embedding-0.6B), wait /health  →  LOCAL ENDPOINT
   4. agent.py       FireworksDeepResearchAgent over subset → run_*.json  (run-file)
                     [or type=external: use your agent's run_*.json directly]
   5. judge.py       Fireworks LLM-as-judge over runs → *_eval.json + summary + CSV
   6. report.py      markdown report (overall + per-domain + per-query)
                                                                    │
 compare.py ─► baseline summary vs candidate summary → deltas + flips + verdict
```

Stage boundaries are files on disk, so any stage can be re-run or replaced independently,
and **your own agent** plugs in at the run-file boundary regardless of language.

## 6. Data contracts (concrete I/O formats)

### 6.1 Run file (agent output — the universal contract)
Per query, `runs/<exp>/run_<ts>_<qid>.json`:
```json
{
  "schema_version": "1.0",
  "query_id": "769",
  "status": "completed",
  "retrieved_docids": ["18639", "41759"],
  "tool_call_counts": {"search": 7, "get_document": 0},
  "result": [
    {"type":"tool_call","tool_name":"search","arguments":"{\"query\":\"...\"}","output":"[...]"},
    {"type":"output_text","output":"Explanation: ... [18639]\nExact Answer: X\nConfidence: 90%"}
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
  "metadata": {"model":"accounts/fireworks/models/gpt-oss-120b","data_source":"...","experiment":"..."}
}
```
**Hard requirements** (consumed by judge): `status=="completed"` for success; the **last**
`result` item is `{"type":"output_text","output": <final answer>}`; `retrieved_docids`,
`tool_call_counts`, `query_id` present; `metadata.model` labels the summary.

### 6.2 Eval file (judge output per query) — mirrors evaluate_run.py
`json_path, query_id, question, response, correct_answer, is_completed, judge_prompt,
judge_response, judge_result{extracted_final_answer, reasoning, correct, confidence,
parse_error}, tool_call_counts, citations{cited_docids, metrics{num_citations,
num_relevant, precision, recall}}, retrieval{retrieved_docids, recall},
model_info{judge_model, max_output_tokens}, domain?`

### 6.3 Summary — leaderboard-compatible + extensions
Original keys: `LLM, Accuracy (%), Recall (%), avg_tool_stats, Calibration Error (%),
Retriever, Link, Evaluation Date, per_query_metrics`. Added: `Citation{...}, num_queries,
num_completed, num_parse_errors, experiment, data_source, agent_model, judge_model,
per_domain{<domain>:{n, accuracy_%, recall_%}}`.

### 6.4 Comparison
`{baseline, candidate, deltas{accuracy_%, recall_%, avg_search_calls, calibration_%},
improvements[{query_id}], regressions[{query_id}], unchanged, per_domain_deltas, verdict}`.

## 7. The three "swap" dimensions → config knobs

- **Model per stage**: `agent.model`, `judge.model` (+ optional `judge.cross_judge_model`).
- **Prompt**: `agent.query_template` (built-in templates) + `agent.system_prompt`.
- **Data source**: `data_source.{backend,index_path,model_name,k,snippet_max_tokens,
  get_document}`. Summary mode = small `snippet_max_tokens`; full-text mode =
  `snippet_max_tokens:-1` and/or `get_document:true`. Web sources = future custom backend
  behind the same retrieval endpoint contract.

## 8. Metric definitions (mapped to paper §4.4)

- **Accuracy** = fraction judged `correct: yes` by LLM-as-judge (semantic equivalence).
- **Recall** = mean over queries of `|retrieved ∩ evidence_qrels| / |evidence_qrels|`.
- **Search Calls** = mean tool calls/query (`avg_tool_stats`).
- **Calibration Error** = RMS calibration of stated confidence vs correctness (needs ≥100
  queries; reported as 0 for tiny smoke runs).
- **Citation P/R** = cited docids vs evidence qrels.

## 9. E2E smoke test

`fireworks_eval/smoke_test.py`:
1. Build a 3-query config (default ids 769/770/771; or `--sample 3 --seed`).
2. Run the full pipeline (managed retrieval endpoint + agent + judge + report).
3. **Assert format + sanity**: run files valid & last item is output_text; completed runs
   have non-empty answer + ≥1 search call + ≥1 retrieved doc; eval files valid; summary
   has Accuracy/Recall in [0,100], `per_query_metrics` length == #queries, no parse errors
   on completed runs.
4. Print PASS/FAIL with diagnostics.

Offline tests (`tests/`, pytest, no network/GPU) validate scoring, run-file schema,
config, and the judge aggregation on fixtures (judge client monkeypatched).

## 10. Module layout (many small, focused files)

`fireworks_eval/`: `config.py, dataset.py, fireworks_client.py, retrieval_server.py,
retrieval_client.py, agent_protocol.py, agent.py, scoring.py, judge.py, report.py,
compare.py, tagging.py, pipeline.py, run_experiment.py, smoke_test.py, errors.py`.
`configs/`: `experiment.example.yaml, smoke.yaml, taxonomy/default.json`.
`tests/`: scoring, agent_protocol, config, offline-smoke + fixtures.
`docs/fireworks_eval.md`: full I/O spec + usage.

## 11. Risks & mitigations

- **gpt-oss tool-call surface via Fireworks chat.completions** — implement defensively
  (handle tool_calls + content), retry transient errors; validated by live smoke.
- **Mac CPU dtype** — default `torch_dtype=auto` → float32 on CPU.
- **2.1GB ground-truth load** — build a slim `{qid,query,answer}` cache once.
- **Self-judging bias** — documented; `judge.model` swappable; optional cross-judge.
- **Upstream sync** — pipeline is self-contained; upstream files untouched.
