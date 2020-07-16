import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from feature_union import FeatureUnion
import numpy as np
from numpy.testing import assert_array_equal

class TestFeatureUnion(unittest.TestCase):
    def setUp(self):
        texts = [
            "私は私のことが好きなあなたを愛しています",
            "私はラーメンが好きです",
            "富士山は日本一高い山です",
        ]
        self.feature_union = FeatureUnion(texts, (2, 2))

    def test_word_bow(self):
        assert_array_equal(
            np.array([
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
            ]), self.feature_union.word_bow_array()
        )

    def test_bigram_bow(self):
        assert_array_equal(
            np.array([
                [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]
            ]), self.feature_union.char_bigram_bow_array()
        )

    def test_unite_feature(self):
        word_bow = self.feature_union.word_bow_array()
        char_bigram_bow = self.feature_union.char_bigram_bow_array()
        assert_array_equal(np.hstack((word_bow, char_bigram_bow)), self.feature_union.unite_feature())

if __name__ == "__main__":
    unittest.main()
