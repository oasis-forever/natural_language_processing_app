import unittest
import sys
sys.path.append("../lib")
from analogy_task import AnalogyTask
import numpy as np
from numpy.testing import assert_almost_equal

class TestAnalogyTask(unittest.TestCase):
    def setUp(self):
        self.analogy_task = AnalogyTask()
        self.analogy_task.word_embeddings()
        self.analogy_task.calc_vector()

    def test_tokyo_array(self):
        assert_almost_equal(np.array(
            [-0.31168, 0.19471, 0.19075, 0.68413, 0.29163, -0.8988, 0.22633, 0.17832, -1.4774, -0.091882, 0.089789, -0.94473, -0.19385, 0.58078, 0.20208, 0.9924, -1.0311, 0.42467, -1.142, 0.71974, 2.1561, -0.14197, -0.92983, -0.28101, -0.011046, -1.6787, 0.44449, 0.54703, -0.71357, -0.67743, 2.3393, 0.28577, 1.4062, -0.0078203, -0.15283, -1.1147, 0.2415, -0.65908, -0.044945, 0.046839, -1.1396, -0.44836, 0.91807, -0.74048, 1.0508, 0.052699, 0.13431, 0.62261, 0.61384, -0.097283]), self.analogy_task.tokyo
        )

    def test_japan_array(self):
        assert_almost_equal(np.array(
            [-0.31739, -0.14033, 0.32292, 1.072, 0.33008, 0.39406, -0.016682, 0.076903, -0.74591, -0.31521, 1.0033, -0.12659, 0.063252, 0.64006, 0.70721, 0.84303, -0.68832, 0.47214, -0.66002, 0.73962, 1.1116, -0.89428, -0.90364, -0.47281, 0.88529, -2.0194, 0.30623, -0.31662, -0.44423, -0.52139, 3.0287, 0.70315, 0.92315, 0.52263, -0.62674, -0.58995, -0.15876, -0.078332, -1.0794, -0.71552, -1.2764, -0.85554, 1.2827, -1.2134, 1.0125, 0.40329, -0.16276, 0.99117, 0.031016, -0.35431]), self.analogy_task.japan
        )

    def test_france_array(self):
        assert_almost_equal(np.array(
            [6.6571e-01, 2.9845e-01, -1.0467e+00, -6.6932e-01, -7.8082e-01, -1.3007e-04, -1.7931e-01, 3.7110e-01, -1.8622e-01, -4.0535e-01, 9.8644e-01, -6.0545e-01, -9.4571e-01, -6.9207e-01, 5.6681e-01, -3.8610e-01, 2.7634e-02, -1.2464e+00, -7.3561e-01, -5.2222e-01, -6.1766e-02, 1.6771e-01, -3.7462e-01, 4.2250e-01, -6.3095e-01, -1.6360e+00, -2.5094e-01, 4.4950e-02, -3.9758e-01, 9.8099e-01, 2.6293e+00, 8.3480e-01, -7.7338e-01, 3.9402e-01, -5.7976e-01, -1.0290e+00, -2.6709e-01, 9.8714e-01, -5.1029e-01, -4.2477e-01, 1.3956e+00, -2.9347e-02, 2.2295e+00, -1.7079e+00, 2.5562e-02, 6.9060e-01, -5.7900e-01, -1.7824e-01, 4.2916e-01, -5.3940e-01]), self.analogy_task.france
        )

    def test_tokyo_shape(self):
        self.assertEqual((50, ), self.analogy_task.tokyo.shape)

    def test_japan_shape(self):
        self.assertEqual((50, ), self.analogy_task.japan.shape)

    def test_france_shape(self):
        self.assertEqual((50, ), self.analogy_task.france.shape)

    def test_closest_word(self):
        self.assertEqual(('paris', 0.9174968004226685), self.analogy_task.closest_word())

if __name__ == "__main__":
    unittest.main()
