import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from latent_semantic_analysis import LatentSemanticAnalysis
import numpy as np
from numpy.testing import assert_almost_equal
import contextlib

class TestLatentSemanticAnalysis(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "車は速く走る",
            "バイクは速く走る",
            "自転車はゆっくり走る",
            "三輪車はゆっくり走る",
            "プログラミングは楽しい",
            "Pythonは楽しい",
        ]
        self.lsa = LatentSemanticAnalysis()

    def _calFUT1(self):
        self.lsa.vectorize(self.texts)
        return self.lsa.bow_shape()

    def _calFUT2(self):
        self.lsa.vectorize(self.texts)
        return self.lsa.bow_table()

    def _calFUT3(self):
        self.lsa.vectorize(self.texts)
        self.lsa.execute_svd()
        return self.lsa.svd_shape()

    def test_bow_shape(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT1()

        actual = buf.getvalue()
        self.assertEqual("Shape: (6, 10)\n", actual)

    def test_bow_table(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT2()

        actual = buf.getvalue()
        self.assertEqual("   python  ゆっくり  バイク  プログラミング  三輪車  楽しい  自転車  走る  車  速い\n0       0     0    0        0    0    0    0   1  1   1\n1       0     0    1        0    0    0    0   1  0   1\n2       0     1    0        0    0    0    1   1  0   0\n3       0     1    0        0    1    0    0   1  0   0\n4       0     0    0        1    0    1    0   0  0   0\n5       1     0    0        0    0    1    0   0  0   0\n", actual)

    def test_svd_shape(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT3()

        actual = buf.getvalue()
        self.assertEqual("Shape: (6, 4)\n", actual)

    def test_svd_array(self):
        self.lsa.vectorize(self.texts)
        self.lsa.execute_svd()
        assert_almost_equal(
            np.array([
                [1.32287566e+00, -2.56314589e-17,  8.66025404e-01, -7.07106781e-01],
                [1.32287566e+00,  1.45849832e-16,  8.66025404e-01,  7.07106781e-01],
                [1.32287566e+00,  5.12279988e-16, -8.66025404e-01, -5.57512786e-16],
                [1.32287566e+00,  3.57744397e-16, -8.66025404e-01, -4.79610367e-16],
                [9.77802511e-17,  1.22474487e+00,  5.98241110e-17,  1.18755510e-16],
                [4.14804552e-18,  1.22474487e+00,  9.71342341e-17,  1.24956643e-16],
            ]), self.lsa.svd_array()
        )

if __name__ == "__main__":
    unittest.main()
