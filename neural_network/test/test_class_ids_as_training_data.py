import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from class_ids_as_training_data import ClassIdsAsTrainingData
import numpy as np
from numpy.testing import assert_array_equal

class TestClassIdsAsTrainingData(unittest.TestCase):
    def setUp(self):
        self.ct = ClassIdsAsTrainingData()

    def test_convert_to_categorical(self):
        assert_array_equal(
            np.array([
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.]
            ]), self.ct.convert_to_categorical()
        )

if __name__ == "__main__":
    unittest.main()
