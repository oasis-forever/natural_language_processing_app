import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from principal_component_analysis import PrincipalComponentAnalysis
import numpy as np
from numpy.testing import assert_almost_equal

class TestPrincipalComponentAnalysis(unittest.TestCase):
    def setUp(self):
        self.pca = PrincipalComponentAnalysis()

    def _calc_cumulative_contribution_ratio(self, ccr):
        return ccr[:2].sum() / ccr.sum()

    def test_shape_features(self):
        self.assertEqual((100, 3), self.pca.shape_features("../npy/sample_features.npy"))

    def test_shape_decomposed_features(self):
        self.pca.shape_features("../npy/sample_features.npy")
        self.assertEqual((100, 2), self.pca.shape_decomposed_features(2))

    def test_explained_variance_ratio(self):
        self.pca.shape_features("../npy/sample_features.npy")
        self.pca.shape_decomposed_features(3)
        assert_almost_equal(np.array([0.82206492, 0.15820158, 0.0197335]), self.pca.explain_variance_ratio())

    def test_shape_decomposed_features_calc_by_cumulative_contribution_ratio(self):
        self.pca.shape_features("../npy/sample_features.npy")
        self.pca.shape_decomposed_features(3)
        ccr = self.pca.explain_variance_ratio()
        dim = self._calc_cumulative_contribution_ratio(ccr)
        self.assertEqual((100, 2), self.pca.shape_decomposed_features(dim))

if __name__ == "__main__":
    unittest.main()
