import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
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
        self.lsa.vectorize(self.texts)

    def _calFUT(self):
        return self.lsa.bow_table()

    def test_bow_shape(self):
        self.assertEqual((6, 10), self.lsa.bow_shape())

    def test_bow_table(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT()

        actual = buf.getvalue()
        self.assertEqual("   python  ゆっくり  バイク  プログラミング  三輪車  楽しい  自転車  走る  車  速い\n0       0     0    0        0    0    0    0   1  1   1\n1       0     0    1        0    0    0    0   1  0   1\n2       0     1    0        0    0    0    1   1  0   0\n3       0     1    0        0    1    0    0   1  0   0\n4       0     0    0        1    0    1    0   0  0   0\n5       1     0    0        0    0    1    0   0  0   0\n", actual)

    def test_svd_shape(self):
        self.lsa.execute_svd()
        self.assertEqual((6, 4), self.lsa.svd_shape())

    def test_svd_array(self):
        self.lsa.execute_svd()
        assert_almost_equal(
            np.array([
                [ 1.32287566e+00, -8.35162776e-18,  8.66025404e-01, -3.37344776e-17],
                [ 1.32287566e+00,  2.16243133e-16,  8.66025404e-01, -8.50948250e-17],
                [ 1.32287566e+00, -2.52539080e-16, -8.66025404e-01, -6.18697181e-01],
                [ 1.32287566e+00,  2.81696359e-16, -8.66025404e-01,  6.18697181e-01],
                [ 9.55967946e-17,  1.22474487e+00, -6.76690461e-17,  3.42365007e-01],
                [-1.50836172e-17,  1.22474487e+00, -1.91520264e-16, -3.42365007e-01]
            ]), self.lsa.svd_array()
        )

if __name__ == "__main__":
    unittest.main()
