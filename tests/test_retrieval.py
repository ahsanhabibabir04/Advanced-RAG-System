
import unittest

from retrieval import HybridRetriever, split_text, tokenize


class TestHybridRetrieval(unittest.TestCase):

    def test_tokenize_lowercases_text(self):
        self.assertEqual(tokenize("Python SEARCH"), ["python", "search"])

    def test_split_text_empty_input(self):
        self.assertEqual(split_text(""), [])

    def test_split_text_creates_overlapping_chunks(self):
        chunks = split_text("abcdefghij", chunk_size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("hello", chunk_size=5, overlap=5)

    def test_empty_retriever_returns_no_results(self):
        retriever = HybridRetriever()
        retriever.fit([])
        self.assertEqual(retriever.search("test"), [])

    def test_fit_rejects_empty_chunk(self):
        retriever = HybridRetriever()
        with self.assertRaises(ValueError):
            retriever.fit(["valid text", "  "])

    def test_search_rejects_empty_query(self):
        retriever = HybridRetriever()
        with self.assertRaises(ValueError):
            retriever.search("  ")

    def test_search_rejects_invalid_top_k(self):
        retriever = HybridRetriever()
        with self.assertRaises(ValueError):
            retriever.search("question", top_k=0)


if __name__ == "__main__":
    unittest.main()
