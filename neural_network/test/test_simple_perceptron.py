import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from simple_perceptron import SimplePerceptron

class TestSimplePerceptron(unittest.TestCase):
    def setUp(self):
        self.sp = SimplePerceptron()
        self.weight = [-1.64, -0.98, 1.31]
        self.bias = -0.05

    def test_perception_1(self):
        x = [0.2, 0.3, -0.1]
        self.assertEqual(0.3093841557917043, self.sp.perception(self.weight, self.bias, x))

    def test_perception_2(self):
        x = [-0.2, -0.1, 0.9]
        self.assertEqual(0.8256347143825868, self.sp.perception(self.weight, self.bias, x))

if __name__ == "__main__":
    unittest.main()
