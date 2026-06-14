import unittest

import numpy as np

from searcher.searchers import SearcherType
from searcher.searchers.openai_embedding_faiss_searcher import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    normalize_query_vector,
    resolve_api_key,
)


class OpenAIEmbeddingFaissSearcherTests(unittest.TestCase):
    def test_searcher_type_registers_openai_embedding_faiss(self):
        self.assertIn("openai-embedding-faiss", SearcherType.get_choices())

    def test_defaults_target_local_qwen3_embedding_endpoint(self):
        self.assertEqual(DEFAULT_EMBEDDING_BASE_URL, "http://localhost:8000/v1")
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "Qwen/Qwen3-Embedding-0.6B")

    def test_resolve_api_key_uses_literal_key_without_env(self):
        self.assertEqual(resolve_api_key(api_key="EMPTY", api_key_env=None), "EMPTY")

    def test_normalize_query_vector_outputs_float32_unit_vector(self):
        vector = normalize_query_vector([3.0, 4.0], normalize=True)

        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (1, 2))
        self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 1.0)

    def test_normalize_query_vector_can_leave_magnitude_unchanged(self):
        vector = normalize_query_vector([3.0, 4.0], normalize=False)

        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (1, 2))
        self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 5.0)


if __name__ == "__main__":
    unittest.main()
