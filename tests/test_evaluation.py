
import unittest

from evaluation import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRetrievalEvaluation(unittest.TestCase):

    def setUp(self):
        self.results = [
            {"chunk_index": 2},
            {"chunk_index": 0},
            {"chunk_index": 3},
            {"chunk_index": 1},
        ]
        self.relevant = {0, 1}

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(self.results, self.relevant), 0.5)

    def test_reciprocal_rank_no_relevant_result(self):
        self.assertEqual(reciprocal_rank(self.results, {99}), 0.0)

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            precision_at_k(self.results, self.relevant, k=2),
            0.5,
        )

    def test_recall_at_k(self):
        self.assertAlmostEqual(
            recall_at_k(self.results, self.relevant, k=2),
            0.5,
        )

    def test_mean_reciprocal_rank(self):
        query_results = [
            (self.results, self.relevant),
            (self.results, {2}),
        ]
        self.assertAlmostEqual(
            mean_reciprocal_rank(query_results),
            0.75,
        )

    def test_empty_query_results(self):
        self.assertEqual(mean_reciprocal_rank([]), 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            precision_at_k(self.results, self.relevant, k=0)


if __name__ == "__main__":
    unittest.main()
