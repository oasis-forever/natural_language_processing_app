import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from multi_layered_perceptron_numpy import MultiLayeredPerceptronNumPy
import numpy as np
from numpy.testing import assert_almost_equal

class TestMultiLayeredPerceptronNumPy(unittest.TestCase):
    def setUp(self):
        self.mlp_numpy = MultiLayeredPerceptronNumPy()
        self.x = np.array([0.2, 0.4, -0.1])
        self.w_1 = np.array([
            [-0.423, -0.795, 1.223],
            [1.402, 0.885, -1.701]
        ])
        self.b_1 = np.array([0.546, 0.774])
        self.w_2 = np.array([
            [1.567, -1.645]
        ])
        self.b_2 = np.array([0.255])

    def test_layer_1(self):
        assert_almost_equal(np.array([0.0211, 1.5785]), self.mlp_numpy.layer_1(self.w_1, self.b_1, self.x))

    def test_layer_2(self):
        self.mlp_numpy.layer_1(self.w_1, self.b_1, self.x)
        assert_almost_equal(np.array([0.0904158]), self.mlp_numpy.layer_2(self.w_2, self.b_2))

if __name__ == "__main__":
    unittest.main()
