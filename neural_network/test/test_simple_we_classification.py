import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from simple_we_classification import SimpleWeClassification
import numpy as np
from numpy.testing import assert_almost_equal

class TestSimpleWeClassification(unittest.TestCase):
    def setUp(self):
        self.swc = SimpleWeClassification()
        self.text = "私は日本人です。"

    def test_calc_text_feature(self):
        assert_almost_equal(
            np.array([
                0.00306862,  0.29269216, -0.07211765, -0.15347076,  0.39600406,
               -0.22959855, -0.0423649 ,  0.04056496, -0.32102068, -0.12297448,
               -0.08545314, -0.17772129, -0.17694424,  0.00245759, -0.40324691,
                0.01976618,  0.06451184,  0.52998734, -0.40892094, -0.10899866,
                0.20896244,  0.15666018,  0.44061819, -0.04938496, -0.21619302,
               -0.44685669,  0.20680416,  0.0726665 ,  0.03446519,  0.12169129,
                0.23724482, -0.34407106, -0.04429496,  0.14432395, -0.24660996,
               -0.18474992,  0.19762295,  0.05516753,  0.195557  , -0.13886511,
                0.01538607,  0.02599786,  0.58125412, -0.06809567,  0.52056654,
                0.09877354,  0.04565307,  0.5780755 , -0.00299348, -0.24687796]
            ), self.swc.calc_text_feature(self.text)
        )

if __name__ == "__main__":
    unittest.main()
