import unittest
import sys
sys.path.append("../lib")
from multi_layer_perceptron_numpy import MultiLayerPerceptronNumPy
import numpy as np
from numpy.testing import assert_almost_equal

class TestMultiLayerPerceptronNumPy(unittest.TestCase):
    def setUp(self):
        self.m_numpy = MultiLayerPerceptronNumPy()

    def test_layer_1(self):
        w_1 = np.array([
            [-0.423, -0.795, 1.223],
            [1.402, 0.885, -1.701]
        ])
        b_1 = np.array([0.546, 0.774])
        x = np.array([0.2, 0.4, -0.1])
        assert_almost_equal(np.array([0.0211, 1.5785]), self.m_numpy.layer_1(w_1, b_1, x))

if __name__ == "__main__":
    unittest.main()
