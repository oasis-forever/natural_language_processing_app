import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from sklearn_tfidf import SkLearnTfIdf
import numpy as np
from numpy.testing import assert_almost_equal

class TestSkLearnTfIdf(unittest.TestCase):
    def setUp(self):
        self.sklearn_tfidf = SkLearnTfIdf()
        self.texts = [
            "私は私のことが好きなあなたを愛しています",
            "私はラーメンが好きです",
            "富士山は日本一高い山です",
        ]

    def test_calc_tfidf(self):
        assert_almost_equal(
            np.array([
                [0.38091445, 0.38091445, 0.38091445, 0., 0.28969526, 0., 0., 0.38091445, 0., 0.57939052, 0.],
                [0., 0., 0., 0.68091856, 0.51785612, 0., 0., 0., 0., 0.51785612, 0.],
                [0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.5, 0., 0.5]
            ]), self.sklearn_tfidf.calc_tfidf(self.texts)
        )

if __name__ == "__main__":
    unittest.main()
