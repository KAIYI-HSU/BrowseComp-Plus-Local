import json
import tempfile
import unittest
from pathlib import Path

from scripts_evaluation.pipeline import _merge_judge_config, build_comparison, load_manifest


class EvaluationPipelineTests(unittest.TestCase):
    def test_default_judge_token_budget_is_large_enough_for_reasoning_model(self):
        config = _merge_judge_config({}, {})

        self.assertEqual(config["max_output_tokens"], 10000)

    def test_load_manifest_requires_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"eval_dir": "evals"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenarios"):
                load_manifest(path)

    def test_build_comparison_computes_deltas_from_first_scenario(self):
        comparison = build_comparison(
            [
                {
                    "name": "baseline",
                    "run_dir": "runs/baseline",
                    "summary_path": "evals/baseline/evaluation_summary.json",
                    "summary": {
                        "Accuracy (%)": 40.0,
                        "Recall (%)": 50.0,
                        "Calibration Error (%)": 20.0,
                        "avg_tool_stats": {"search": 10.0},
                    },
                },
                {
                    "name": "candidate",
                    "run_dir": "runs/candidate",
                    "summary_path": "evals/candidate/evaluation_summary.json",
                    "summary": {
                        "Accuracy (%)": 45.5,
                        "Recall (%)": 55.0,
                        "Calibration Error (%)": 18.0,
                        "avg_tool_stats": {"search": 8.5},
                    },
                },
            ]
        )

        self.assertEqual(comparison["baseline"], "baseline")
        self.assertEqual(comparison["scenarios"][0]["delta_accuracy"], 0.0)
        self.assertEqual(comparison["scenarios"][1]["delta_accuracy"], 5.5)
        self.assertEqual(comparison["scenarios"][1]["delta_recall"], 5.0)
        self.assertEqual(comparison["scenarios"][1]["delta_search_calls"], -1.5)
        self.assertEqual(comparison["scenarios"][1]["delta_calibration_error"], -2.0)


if __name__ == "__main__":
    unittest.main()
