import unittest
import sys
sys.path.append("../eval")
import classification_report as cr

class TestPrecisionRecall(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 2, 1, 0, 0, 1]

    def _calFUT(self):
        return cr.print_report(self.y_true, self.y_pred)

    def test_eval_each_class_precision(self):
        self.assertEqual('              precision    recall  f1-score   support\n\n           0       0.67      1.00      0.80         2\n           1       0.00      0.00      0.00         2\n           2       0.00      0.00      0.00         2\n\n    accuracy                           0.33         6\n   macro avg       0.22      0.33      0.27         6\nweighted avg       0.22      0.33      0.27         6\n', cr.report(self.y_true, self.y_pred))

if __name__ == "__main__":
    unittest.main()
