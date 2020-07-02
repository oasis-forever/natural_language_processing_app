import unittest
import sys
sys.path.append("../lib")
from vector import Vector
import numpy as np

class TestVector(unittest.TestCase):
    def setUp(self):
        self.vector = Vector(np.array([0, 1, 2, 3, 4, 5]))

    def test_array(self):
        np.alltrue(np.array([0, 1, 2, 3, 4, 5]) == self.vector.array)

    def test_shape(self):
        self.assertEqual((6,), self.vector.shape)

    def test_slice(self):
        np.alltrue(np.array([0, 1, 2]) == self.vector.array[:3])

    def test_max(self):
        self.assertEqual(5, self.vector.max)

    def test_min(self):
        self.assertEqual(0, self.vector.min)

    def test_argmax(self):
        self.assertEqual(5, self.vector.argmax)

    def test_argmin(self):
        self.assertEqual(0, self.vector.argmin)

    def test_sum(self):
        array = np.array([6, 7, 8, 9, 10, 11])
        np.alltrue(np.array([6, 8, 10, 12, 14, 16]) == self.vector.sum(array))

    def test_multiply(self):
        array = np.array([6, 7, 8, 9, 10, 11])
        np.alltrue(np.array([0, 7, 16, 27, 40, 55]) == self.vector.multiply(array))

    def test_dot_product(self):
        array = np.array([6, 7, 8, 9, 10, 11])
        self.assertEqual(145, self.vector.dot_product(array))

if __name__ == "__main__":
    unittest.main()
