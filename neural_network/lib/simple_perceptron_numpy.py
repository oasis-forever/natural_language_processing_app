import numpy as np

class SimplePerceptronNumPy:
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def perception(self, weight, bias, x):
        z = bias + np.dot(x, weight)
        output = self._sigmoid(z)
        return output
