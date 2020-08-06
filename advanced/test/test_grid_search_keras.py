import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from grid_search_keras import GridSearchKeras
import numpy as np
from numpy.testing import assert_almost_equal
import contextlib

class TestGridSearchKeras(unittest.TestCase):
    def setUp(self):
        self.gsk = GridSearchKeras()
        self.gsk.feature_extraction()
        self.gsk.listup_params()

    def _calFUT(self):
        return self.gsk.search_best_params()

    def test_search_best_params(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT()

        actual = buf.getvalue()
        self.assertEqual("{"batch_size": 16, "dropout": 0, "epochs": 10, "learning_rate": 0.1, "optimizer_class": <class "tensorflow.python.keras.optimizer_v2.gradient_descent.SGD">}", actual)

    # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/4
    def test_classify_with_best_params(self):
        self.gsk.search_best_params()
        assert_almost_equal(
            np.array([]), self.gsk.classify_with_best_params()
        )

if __name__ == "__main__":
    unittest.main()
