import unittest
from unittest.mock import patch

import numpy as np

from src.retrieval import hybrid_search


class HybridSearchTest(unittest.TestCase):
    def test_item_found_by_both_signals_ranks_first(self):
        # "b" ranks #2 in vector search and #1 in BM25 — RRF should put it
        # above "a" (vector-only, rank #1) and "c" (BM25-only, rank #2).
        with (
            patch("src.retrieval.embed_query", return_value=np.zeros(4, dtype=np.float32)),
            patch(
                "src.retrieval.vectorstore.query",
                return_value=[
                    {"id": "a", "text": "a", "metadata": {}},
                    {"id": "b", "text": "b", "metadata": {}},
                ],
            ),
            patch(
                "src.retrieval.bm25_index.query",
                return_value=[("b", 5.0), ("c", 3.0)],
            ),
            patch(
                "src.retrieval.vectorstore.get_by_ids",
                return_value={
                    "a": {"text": "doc a", "metadata": {"source": "a.pdf"}},
                    "b": {"text": "doc b", "metadata": {"source": "b.pdf"}},
                    "c": {"text": "doc c", "metadata": {"source": "c.pdf"}},
                },
            ),
        ):
            results = hybrid_search("test query", top_k=5)

        self.assertEqual([r["id"] for r in results], ["b", "a", "c"])

    def test_no_hits_returns_empty_list(self):
        with (
            patch("src.retrieval.embed_query", return_value=np.zeros(4, dtype=np.float32)),
            patch("src.retrieval.vectorstore.query", return_value=[]),
            patch("src.retrieval.bm25_index.query", return_value=[]),
            patch("src.retrieval.vectorstore.get_by_ids", return_value={}),
        ):
            results = hybrid_search("nothing matches")

        self.assertEqual(results, [])

    def test_respects_top_k(self):
        vector_hits = [{"id": f"v{i}", "text": "", "metadata": {}} for i in range(5)]
        documents = {f"v{i}": {"text": "", "metadata": {}} for i in range(5)}
        with (
            patch("src.retrieval.embed_query", return_value=np.zeros(4, dtype=np.float32)),
            patch("src.retrieval.vectorstore.query", return_value=vector_hits),
            patch("src.retrieval.bm25_index.query", return_value=[]),
            patch("src.retrieval.vectorstore.get_by_ids", return_value=documents),
        ):
            results = hybrid_search("test query", top_k=2)

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
