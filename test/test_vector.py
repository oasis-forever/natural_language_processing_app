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


if __name__ == "__main__":
    unittest.main()
