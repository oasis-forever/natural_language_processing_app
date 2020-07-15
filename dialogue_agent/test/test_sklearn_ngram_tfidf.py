import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from sklearn_ngram_tfidf import SkLearnNgramTfIdf
import numpy as np
from numpy.testing import assert_almost_equal

class TestSkLearnNgramTfIdf(unittest.TestCase):
    def setUp(self):
        texts = ["東京から大阪に行く", "大阪から東京に行く"]
        self.sklearn_ngram_tfidf = SkLearnNgramTfIdf(texts, (2, 2))

    def test_bow(self):
        assert_almost_equal(
            np.array(
                [
                    [0.53404633, 0., 0.37997836, 0., 0.53404633, 0.53404633, 0.],
                    [0, 0.53404633, 0.37997836, 0.53404633, 0., 0., 0.53404633]
                ]
            ),
            self.sklearn_ngram_tfidf.bow_array()
        )

    def test_vocabulary(self):
        self.assertEqual({"東京 から": 5, "から 大阪": 0, "大阪 に": 4, "に 行く": 2, "大阪 から": 3, "から 東京": 1, "東京 に": 6}, self.sklearn_ngram_tfidf.vocabulary())

if __name__ == "__main__":
    unittest.main()
