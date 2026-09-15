import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class IngestReplaceTest(unittest.TestCase):
    def test_reingest_same_filename_replaces_old_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vector_path = tmp_path / "vector_store.pkl"
            bm25_path = tmp_path / "bm25_index.pkl"

            with (
                patch("src.vectorstore.VECTOR_STORE_PATH", vector_path),
                patch("src.bm25_index.BM25_INDEX_PATH", bm25_path),
                patch(
                    "src.ingest_pipeline.embed_passages",
                    side_effect=lambda texts: np.eye(len(texts), 4, dtype=np.float32),
                ),
                patch("src.ingest_pipeline.load_file") as load_file,
                patch("src.ingest_pipeline.chunk_text") as chunk_text,
            ):
                from src.ingest_pipeline import ingest_file
                from src import bm25_index, vectorstore

                path = Path("report.pdf")
                load_file.return_value = "old"
                chunk_text.return_value = ["old chunk a", "old chunk b"]
                self.assertEqual(ingest_file(path), 2)

                load_file.return_value = "new"
                chunk_text.return_value = ["new chunk only"]
                self.assertEqual(ingest_file(path), 1)

                store = vectorstore._load()
                self.assertEqual(len(store["ids"]), 1)
                self.assertEqual(store["texts"], ["new chunk only"])
                self.assertEqual(store["embeddings"].shape[0], 1)

                bm25 = bm25_index._load()
                self.assertEqual(len(bm25["ids"]), 1)
                self.assertEqual(bm25["ids"], store["ids"])


if __name__ == "__main__":
    unittest.main()
