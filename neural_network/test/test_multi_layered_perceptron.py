import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from multi_layered_perceptron import MultiLayeredPerceptron

class TestMultiLayeredPerceptron(unittest.TestCase):
    def setUp(self):
        self.mlp = MultiLayeredPerceptron()
        self.weights = []
        self.biases = []

    def test_layer_1(self):
        w_11 = [-0.423, -0.795, 1.223]
        self.weights.append(w_11)
        b_11 = 0.546
        self.biases.append(b_11)
        w_12 = [1.402, 0.885, -1.701]
        self.weights.append(w_12)
        b_12 = 0.774
        self.biases.append(b_12)
        x = [0.2, 0.4, -0.1]
        self.assertEqual([0.021099999999999952, 1.5785], self.mlp.layer_1(self.weights, self.biases, x))

if __name__ == "__main__":
    unittest.main()
