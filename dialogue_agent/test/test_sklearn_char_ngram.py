import unittest
import sys
sys.path.append("../lib")
from sklearn_char_ngram import SkLearnCharNgram
import numpy as np
from numpy.testing import assert_array_equal

class TestSkLearnCharNgram(unittest.TestCase):
    def setUp(self):
        texts = ["東京から大阪に行く", "大阪から東京に行く"]
        self.sklearn_ngram = SkLearnCharNgram(texts, (3, 3))

    def test_bow(self):
        assert_array_equal(
            np.array(
                [
                    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
                ]
            ),
            self.sklearn_ngram.bow_array()
        )

    def test_vocabulary(self):
        self.assertEqual({"東京か": 9, "京から": 5, "から大": 0, "ら大阪": 3, "大阪に": 8, "阪に行": 12, "に行く": 2, "大阪か": 7, "阪から": 11, "から東": 1, "ら東京": 4, "東京に": 10, "京に行": 6}, self.sklearn_ngram.vocabulary())

if __name__ == "__main__":
    unittest.main()
