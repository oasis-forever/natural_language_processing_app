import unittest
import sys
sys.path.append("../eval")
import precision_recall as pr
import numpy as np
from numpy.testing import assert_almost_equal

class TestPrecisionRecall(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 2, 1, 0, 0, 1]

    def test_eval_each_class_precision(self):
        assert_almost_equal(np.array([0.66666667, 0., 0.]), pr.eval_each_class_precision(self.y_true, self.y_pred))

    def test_eval_avarage_precision(self):
        self.assertEqual(0.2222222222222222, pr.eval_average_precision(self.y_true, self.y_pred))

    def test_eval_each_class_recall(self):
        assert_almost_equal(np.array([1., 0., 0.]), pr.eval_each_class_recall(self.y_true, self.y_pred))

    def test_eval_average_recall(self):
        self.assertEqual(0.3333333333333333, pr.eval_average_recall(self.y_true, self.y_pred))

if __name__ == "__main__":
    unittest.main()
