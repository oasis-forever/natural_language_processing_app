import numpy as np
import sys
sys.path.append("../concern")
from numpy_perceptron import sigmoid
from numpy_perceptron import rectified_liner_unit
from numpy_perceptron import calc_innner_product

class MultiLayeredPerceptronNumPy:
    def __init__(self):
        pass

    def layer_1(self, weights, bias, x):
        z = calc_innner_product(weights, bias, x)
        self.out_1 = rectified_liner_unit(z)
        return self.out_1

    def layer_2(self, weights, bias):
        z = calc_innner_product(weights, bias, self.out_1)
        self.out_2 = sigmoid(z)
        return self.out_2
