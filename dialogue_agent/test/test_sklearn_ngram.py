import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from sklearn_ngram import SkLearnNgram
import numpy as np
from numpy.testing import assert_array_equal

class TestSkLearnNgram(unittest.TestCase):
    def setUp(self):
        texts = ["東京から大阪に行く", "大阪から東京に行く"]
        self.sklearn_ngram = SkLearnNgram(texts, (2, 2))

    def test_bow(self):
        assert_array_equal(
            np.array(
                [
                    [1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 0, 1]
                ]
            ),
            self.sklearn_ngram.bow_array()
        )

    def test_vocabulary(self):
        self.assertEqual({'東京 から': 5, 'から 大阪': 0, '大阪 に': 4, 'に 行く': 2, '大阪 から': 3, 'から 東京': 1, '東京 に': 6}, self.sklearn_ngram.vocabulary())

if __name__ == "__main__":
    unittest.main()
