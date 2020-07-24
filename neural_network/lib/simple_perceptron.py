import math

class SimplePerceptron:
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def perception(self, weight, bias, x):
        z = bias + x[0] * weight[0] + x[1] * weight[1] + x[2] * weight[2]
        output = self._sigmoid(z)
        return output
