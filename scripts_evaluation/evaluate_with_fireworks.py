from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from scripts_evaluation.evaluation_core import evaluate_directory


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS = 10000


@dataclass
class FireworksJudge:
    client: openai.OpenAI
    model: str
    max_output_tokens: int
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None

    def judge(self, prompt: str, *, query_id: str) -> str:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**request)
        if not response.choices:
            return ""
        message = response.choices[0].message
        return message.content or ""


def build_client(*, api_key_env: str, base_url: str) -> openai.OpenAI:
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set in the environment or .env")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BrowseComp-Plus runs with a Fireworks/OpenAI-compatible judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing run JSON files"
    )
    parser.add_argument(
        "--ground_truth",
        default="data/browsecomp_plus_decrypted.jsonl",
        help="Path to decrypted BrowseComp-Plus JSONL ground truth",
    )
    parser.add_argument(
        "--eval_dir", default="./evals", help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--qrel_evidence",
        default="topics-qrels/qrel_evidence.txt",
        help="Path to evidence-document qrels",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_FIREWORKS_JUDGE_MODEL,
        help="Fireworks model path for judging",
    )
    parser.add_argument(
        "--base-url",
        default=FIREWORKS_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="FIREWORKS_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS,
        help="Maximum judge output tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Judge sampling temperature"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional Fireworks reasoning_effort parameter",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-evaluation of existing files"
    )
    args = parser.parse_args()

    client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)
    judge = FireworksJudge(
        client=client,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    report = evaluate_directory(
        input_dir=Path(args.input_dir),
        ground_truth_path=Path(args.ground_truth),
        eval_dir=Path(args.eval_dir),
        qrel_evidence_path=Path(args.qrel_evidence),
        judge=judge,
        judge_model=args.model,
        max_output_tokens=args.max_output_tokens,
        force=args.force,
    )

    print(f"Evaluated {len(report.results)} responses")
    print(f"Accuracy: {report.summary['Accuracy (%)']:.2f}%")
    recall = report.summary.get("Recall (%)")
    print(f"Recall: {recall:.2f}%" if isinstance(recall, (int, float)) else "Recall: N/A")
    print(f"Summary saved to {report.summary_path}")


if __name__ == "__main__":
    main()
