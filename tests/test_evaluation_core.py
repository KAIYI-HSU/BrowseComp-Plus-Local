import json
import tempfile
import unittest
from pathlib import Path

from scripts_evaluation.evaluation_core import (
    evaluate_directory,
    validate_run_record,
)


class StaticJudge:
    def __init__(self, text: str):
        self.text = text
        self.prompts = []

    def judge(self, prompt: str, *, query_id: str) -> str:
        self.prompts.append((query_id, prompt))
        return self.text


class EvaluationCoreTests(unittest.TestCase):
    def test_validate_run_record_reports_contract_errors(self):
        errors = validate_run_record(
            {
                "query_id": 123,
                "status": "completed",
                "tool_call_counts": {"search": "two"},
                "retrieved_docids": ["1"],
                "result": "not-a-list",
            },
            source="bad.json",
        )

        self.assertIn("bad.json: query_id must be a string", errors)
        self.assertIn("bad.json: tool_call_counts.search must be an int", errors)
        self.assertIn("bad.json: result must be a list", errors)

    def test_evaluate_directory_writes_per_query_outputs_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs" / "scenario_a"
            runs.mkdir(parents=True)
            data = root / "data"
            data.mkdir()
            evals = root / "evals"

            ground_truth = data / "browsecomp_plus_decrypted.jsonl"
            ground_truth.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query_id": "q1",
                                "query": "Question one?",
                                "answer": "Alpha",
                            }
                        ),
                        json.dumps(
                            {
                                "query_id": "q2",
                                "query": "Question two?",
                                "answer": "Beta",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            qrels = root / "qrel_evidence.txt"
            qrels.write_text("q1 0 1 1\nq1 0 2 1\nq2 0 3 1\n", encoding="utf-8")

            (runs / "q1.json").write_text(
                json.dumps(
                    {
                        "metadata": {"model": "agent-a"},
                        "query_id": "q1",
                        "tool_call_counts": {"search": 2},
                        "status": "completed",
                        "retrieved_docids": ["1", "99"],
                        "result": [
                            {
                                "type": "output_text",
                                "output": "Explanation: cited [1].\nExact Answer: Alpha\nConfidence: 80%",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (runs / "q2.json").write_text(
                json.dumps(
                    {
                        "metadata": {"model": "agent-a"},
                        "query_id": "q2",
                        "tool_call_counts": {"search": 1},
                        "status": "incomplete",
                        "retrieved_docids": ["3"],
                        "result": [],
                    }
                ),
                encoding="utf-8",
            )

            judge = StaticJudge(
                "\n".join(
                    [
                        "extracted_final_answer: Alpha",
                        "[correct_answer]: Alpha",
                        "reasoning: same answer",
                        "correct: yes",
                        "confidence: 80%",
                    ]
                )
            )

            report = evaluate_directory(
                input_dir=runs,
                ground_truth_path=ground_truth,
                eval_dir=evals,
                qrel_evidence_path=qrels,
                judge=judge,
                judge_model="fake-judge",
                max_output_tokens=128,
                force=True,
            )

            self.assertEqual([qid for qid, _ in judge.prompts], ["q1"])
            self.assertEqual(report.summary["LLM"], "agent-a")
            self.assertEqual(report.summary["Accuracy (%)"], 50.0)
            self.assertEqual(report.summary["Recall (%)"], 75.0)
            self.assertEqual(report.summary["avg_tool_stats"]["search"], 1.5)
            self.assertEqual(report.summary["Completed"], 1)
            self.assertEqual(report.summary["Incomplete"], 1)
            self.assertEqual(report.summary["Parse Errors"], 1)
            self.assertEqual(
                report.summary["per_query_metrics"],
                [
                    {"query_id": "q1", "correct": True, "recall": 50.0},
                    {"query_id": "q2", "correct": False, "recall": 100.0},
                ],
            )

            self.assertTrue((report.output_dir / "q1_eval.json").is_file())
            self.assertTrue((report.output_dir / "q2_eval.json").is_file())
            self.assertTrue((report.output_dir / "evaluation_summary.json").is_file())
            self.assertTrue((report.output_dir / "detailed_judge_results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
