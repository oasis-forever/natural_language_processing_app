import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from sklearn_basic import SkLearnBasic
import numpy as np
from numpy.testing import assert_array_equal

class TestSkLearnBasic(unittest.TestCase):
    def setUp(self):
        self.sklearn_basic = SkLearnBasic()
        self.texts = [
            "私は私のことが好きなあなたを愛しています",
            "私はラーメンが好きです",
            "富士山は日本一高い山です",
        ]

    def test_calc_bow(self):
        assert_array_equal(
            np.array([
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
            ]), self.sklearn_basic.calc_bow(self.texts)
        )

if __name__ == "__main__":
    unittest.main()
