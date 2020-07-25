import unittest
import sys
sys.path.append("../lib")
from sklearn_adhoc_union import TextStats
import numpy as np
from numpy.testing import assert_almost_equal

class TestSkLearnAdhocUnion(unittest.TestCase):
    def setUp(self):
        self.sklearn_adhoc_union = TextStats()

    def test_unite_feature(self):
        texts = [ "こんにちは。こんばんは。", "焼肉が食べたい"]
        assert_almost_equal(
            np.array([
                [12., 2., 1., 0., 2., 0., 1., 1., 2., 1., 0., 1., 1., 1., 0., 0., 0.],
                [7., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1]
            ])
        , self.sklearn_adhoc_union.unite_feature(texts, (2, 2))
        )

if __name__ == "__main__":
    unittest.main()
