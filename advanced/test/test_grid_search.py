import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from grid_search import GridSearch
import numpy as np
from numpy.testing import assert_almost_equal

class TestGridSearch(unittest.TestCase):
    def setUp(self):
        self.grid_search = GridSearch()
        self.grid_search.feature_extraction()

    def test_search_best_params(self):
        self.assertEqual({"max_features": "log2", "n_estimators": 500}, self.grid_search.search_best_params())

    # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/3
    def test_classify_with_best_params(self):
        self.grid_search.search_best_params()
        assert_almost_equal(
            np.array([
                48, 48, 25, 25, 23, 40,  6,  6, 21,  6, 20, 20, 39, 19, 17, 11, 16,
                16, 13, 13, 21, 18, 10, 10, 14,  3,  9,  8,  7,  7,  6,  6,  5,  5,
                8, 45, 21,  3,  2, 46,  1,  1,  6,  1,  0,  0,  6, 26, 41,  6, 38,
                38, 37, 37,  6, 22, 27, 27, 42, 42, 10, 21, 44, 19, 34, 34, 33, 33,
                45, 45, 32, 32, 31, 16, 46, 46, 16, 33, 28, 28,  6, 39, 24, 24, 14,
                14, 15, 15, 18, 15, 45, 30, 12, 35
            ]), self.grid_search.classify_with_best_params()
        )

if __name__ == "__main__":
    unittest.main()
