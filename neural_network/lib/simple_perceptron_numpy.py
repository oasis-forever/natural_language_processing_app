import numpy as np
import sys
sys.path.append("../concern")
from numpy_perceptron import sigmoid
from numpy_perceptron import calc_innner_product

class SimplePerceptronNumPy:
    def __init__(self):
        pass

    def perception(self, weight, bias, x):
        z = calc_innner_product(weight, bias, x)
        output = sigmoid(z)
        return output
