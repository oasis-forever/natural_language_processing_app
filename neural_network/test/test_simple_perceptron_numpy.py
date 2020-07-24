import unittest
import sys
sys.path.append("../lib")
from simple_perceptron_numpy import SimplePerceptronNumPy
import numpy as np

class TestSimplePerceptronNumPy(unittest.TestCase):
    def setUp(self):
        self.s_numpy = SimplePerceptronNumPy()
        self.weight = np.array([-1.64, -0.98, 1.31])
        self.bias = -0.05

    def test_perception_1(self):
        x = [0.2, 0.3, -0.1]
        self.assertEqual(0.3093841557917043, self.s_numpy.perception(self.weight, self.bias, x))

    def test_perception_2(self):
        x = [-0.2, -0.1, 0.9]
        self.assertEqual(0.8256347143825868, self.s_numpy.perception(self.weight, self.bias, x))

if __name__ == "__main__":
    unittest.main()
