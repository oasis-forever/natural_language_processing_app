import numpy as np
import sys
sys.path.append("../concern")
from numpy_perceptron import sigmoid
from numpy_perceptron import rectified_liner_unit
from numpy_perceptron import calc_innner_product

class MultiLayerPerceptronNumPy:
    def __init__(self):
        pass

    def layer_1(self, weights, bias, x):
        z = calc_innner_product(weights, bias, x)
        self.out_1 = rectified_liner_unit(z)
        return self.out_1

