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


if __name__ == "__main__":
    unittest.main()
