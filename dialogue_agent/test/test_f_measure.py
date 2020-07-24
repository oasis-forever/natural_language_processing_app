import unittest
import sys
sys.path.append("../eval")
import f_measure as fm

class TestPrecisionRecall(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 2, 1, 0, 0, 1]

    def test_eval_each_class_precision(self):
        self.assertEqual(0.26666666666666666, fm.eval_f1_score(self.y_true, self.y_pred))

if __name__ == "__main__":
    unittest.main()
