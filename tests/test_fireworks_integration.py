import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts_evaluation.evaluate_with_fireworks import (
    DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS,
    FireworksJudge,
)
from search_agent.fireworks_client import (
    persist_chat_response,
    run_conversation_with_tools,
    to_chat_tool_definitions,
)


class FakeChatCompletions:
    def __init__(self, choices=None):
        self.calls = []
        self.choices = choices or [
            SimpleNamespace(
                message=SimpleNamespace(content="correct: yes\nconfidence: 90%")
            )
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=self.choices)


class FakeOpenAIClient:
    def __init__(self, choices=None):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(choices=choices))


class FireworksIntegrationTests(unittest.TestCase):
    def test_fireworks_judge_default_token_budget_is_large_enough_for_reasoning_model(self):
        self.assertEqual(DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS, 10000)

    def test_chat_completion_request_excludes_metadata_only_fields(self):
        client = FakeOpenAIClient()
        initial_request = {
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "messages": [{"role": "user", "content": "Question?"}],
            "max_tokens": 256,
            "temperature": 0.0,
            "api_base_url": "https://api.fireworks.ai/inference/v1",
        }

        messages, tool_usage, status = run_conversation_with_tools(
            client=client,
            initial_request=initial_request,
            tool_handler=object(),
            max_iterations=1,
        )

        self.assertEqual(status, "completed")
        self.assertEqual(tool_usage, {})
        self.assertEqual(messages[-1]["content"], "correct: yes\nconfidence: 90%")
        call = client.chat.completions.calls[0]
        self.assertNotIn("api_base_url", call)

    def test_reasoning_only_response_without_final_content_is_incomplete(self):
        client = FakeOpenAIClient(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="Still thinking through the answer.",
                    ),
                    finish_reason="stop",
                )
            ]
        )
        initial_request = {
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "messages": [{"role": "user", "content": "Question?"}],
            "max_tokens": 256,
            "temperature": 0.0,
        }

        messages, tool_usage, status = run_conversation_with_tools(
            client=client,
            initial_request=initial_request,
            tool_handler=object(),
            max_iterations=1,
        )

        self.assertEqual(status, "incomplete")
        self.assertEqual(tool_usage, {})
        self.assertEqual(
            messages[-1]["reasoning_content"], "Still thinking through the answer."
        )

    def test_length_limited_response_is_incomplete(self):
        client = FakeOpenAIClient(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Partial final answer"),
                    finish_reason="length",
                )
            ]
        )
        initial_request = {
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "messages": [{"role": "user", "content": "Question?"}],
            "max_tokens": 256,
            "temperature": 0.0,
        }

        _, _, status = run_conversation_with_tools(
            client=client,
            initial_request=initial_request,
            tool_handler=object(),
            max_iterations=1,
        )

        self.assertEqual(status, "incomplete")

    def test_fireworks_judge_uses_chat_completions_request_shape(self):
        client = FakeOpenAIClient()
        judge = FireworksJudge(
            client=client,
            model="accounts/fireworks/models/gpt-oss-120b",
            max_output_tokens=256,
            temperature=0.0,
            reasoning_effort="low",
        )

        text = judge.judge("grade this", query_id="q1")

        self.assertEqual(text, "correct: yes\nconfidence: 90%")
        self.assertEqual(len(client.chat.completions.calls), 1)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "accounts/fireworks/models/gpt-oss-120b")
        self.assertEqual(call["messages"], [{"role": "user", "content": "grade this"}])
        self.assertEqual(call["max_tokens"], 256)
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["reasoning_effort"], "low")

    def test_chat_tool_definition_conversion_and_persisted_run_schema(self):
        response_tools = [
            {
                "type": "function",
                "name": "local_knowledge_base_retrieval",
                "description": "Search docs",
                "parameters": {
                    "type": "object",
                    "properties": {"user_query": {"type": "string"}},
                    "required": ["user_query"],
                },
                "strict": True,
            }
        ]

        self.assertEqual(
            to_chat_tool_definitions(response_tools),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "local_knowledge_base_retrieval",
                        "description": "Search docs",
                        "parameters": {
                            "type": "object",
                            "properties": {"user_query": {"type": "string"}},
                            "required": ["user_query"],
                        },
                        "strict": True,
                    },
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            messages = [
                {"role": "user", "content": "Question?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "local_knowledge_base_retrieval",
                                "arguments": json.dumps({"user_query": "Question?"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": json.dumps([{"docid": "42", "snippet": "answer doc"}]),
                },
                {
                    "role": "assistant",
                    "content": "Explanation: cited [42].\nExact Answer: Answer\nConfidence: 90%",
                },
            ]

            path = persist_chat_response(
                out_dir=out_dir,
                initial_request={
                    "model": "accounts/fireworks/models/gpt-oss-120b",
                    "reasoning_effort": "high",
                },
                messages=messages,
                tool_usage={"local_knowledge_base_retrieval": 1},
                status="completed",
                query_id="q1",
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["query_id"], "q1")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["tool_call_counts"], {"search": 1})
            self.assertEqual(record["retrieved_docids"], ["42"])
            self.assertEqual(record["result"][-1]["type"], "output_text")
            self.assertIn("Exact Answer: Answer", record["result"][-1]["output"])


if __name__ == "__main__":
    unittest.main()
