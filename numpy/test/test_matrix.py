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

    def test_vertica_max(self):
        np.alltrue(np.array([6, 7, 8]) == self.matrix.v_max)

    def test_horizontal_max(self):
        np.alltrue(np.array([2, 5, 8]) == self.matrix.h_max)

    def test_vertica_min(self):
        np.alltrue(np.array([0, 1, 2]) == self.matrix.v_min)

    def test_horizontal_min(self):
        np.alltrue(np.array([0, 3, 6]) == self.matrix.h_min)

    def test_vertica_argmax(self):
        np.alltrue(np.array([2, 2, 2]) == self.matrix.v_argmax)

    def test_horizontal_argmax(self):
        np.alltrue(np.array([2, 2, 2]) == self.matrix.h_argmax)

    def test_vertica_argmin(self):
        np.alltrue(np.array([0, 0, 0]) == self.matrix.v_argmin)

    def test_horizontal_argmin(self):
        np.alltrue(np.array([0, 0, 0]) == self.matrix.h_argmin)

    def test_arr_sum(self):
        array = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        np.alltrue(np.array([[9, 11, 13], [15, 17, 19], [21, 23, 25]]) == self.matrix.sum(array))

    def test_arr_multiply(self):
        array = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        np.alltrue(np.array([[0, 10, 22], [36, 52, 70], [90, 112, 136]]) == self.matrix.multiply(array))

    def test_broadcasting_sum(self):
        np.alltrue(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]) == self.matrix.sum(10))

    def test_broadcasting_multiply(self):
        np.alltrue(np.array([[0, 10, 20], [30, 40, 50], [60, 70, 80]]) == self.matrix.multiply(10))

    def test_dot_product(self):
        array = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])
        np.alltrue(np.array([[42, 45, 48], [150, 162, 174], [258, 279, 300]]) == self.matrix.dot_product(array))

    def test_matrix_multiply_vector(self):
        vector = np.array([9, 8, 7])
        np.alltrue(np.array([22, 94, 166]) == self.matrix.dot_product(vector))

    def test_np_sum(self):
        self.assertEqual(36, self.matrix.np_sum())

    def test_np_mean(self):
        self.assertEqual(4, self.matrix.np_mean())

    def test_exponential(self):
        np.alltrue(np.array([[1.00000000e+00, 2.71828183e+00, 7.38905610e+00], [2.00855369e+01, 5.45981500e+01, 1.48413159e+02], [4.03428793e+02, 1.09663316e+03, 2.98095799e+03]]) == self.matrix.exponential())

if __name__ == "__main__":
    unittest.main()
