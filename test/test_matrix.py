import unittest
import sys
sys.path.append("../lib")
from matrix import Matrix
import numpy as np

class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.matrix = Matrix(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

    def test_array(self):
        np.alltrue(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]) == self.matrix.array)

    def test_shape(self):
        self.assertEqual((3, 3), self.matrix.shape)

    def test_slice(self):
        np.alltrue(np.array([[1, 2], [4, 5]]) == self.matrix.array[:2, 1:])

    def test_arr_sum(self):
        list = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        np.alltrue(np.array([[9, 11, 13], [15, 17, 19], [21, 23, 25]]) == self.matrix.sum(list))

    def test_arr_multiply(self):
        list = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        np.alltrue(np.array([[0, 10, 22], [36, 52, 70], [90, 112, 136]]) == self.matrix.multiply(list))

if __name__ == "__main__":
    unittest.main()
