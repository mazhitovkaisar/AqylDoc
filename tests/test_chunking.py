import unittest

from src.chunking import chunk_text


class ChunkTextTest(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("", chunk_size=10, overlap=2), [])

    def test_whitespace_only_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   \n\t  ", chunk_size=10, overlap=2), [])

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        self.assertEqual(chunk_text("one two three", chunk_size=10, overlap=2), ["one two three"])

    def test_splits_into_overlapping_chunks(self):
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=4, overlap=1)

        self.assertEqual(
            chunks,
            ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"],
        )

    def test_no_overlap_splits_into_disjoint_chunks(self):
        words = [f"w{i}" for i in range(8)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=4, overlap=0)

        self.assertEqual(chunks, ["w0 w1 w2 w3", "w4 w5 w6 w7"])


if __name__ == "__main__":
    unittest.main()
