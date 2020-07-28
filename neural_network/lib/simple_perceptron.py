import math
import sys
sys.path.append("./concern")
from math_perceptron import sigmoid

class SimplePerceptron:
    def __init__(self):
        pass

    def perception(self, weight, bias, x):
        z = bias + x[0] * weight[0] + x[1] * weight[1] + x[2] * weight[2]
        output = sigmoid(z)
        return output
