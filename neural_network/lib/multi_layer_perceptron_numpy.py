import numpy as np

class MultiLayerPerceptronNumPy:
    def __init__(self):
        pass

    def _rectified_liner_unit(self, z):
        return np.maximum(z, 0)

    def _perceptions(self, weights, bias, x):
        z_1 = bias + np.dot(weights, x)
        output1 = self._rectified_liner_unit(z_1)
        return output1

    def layer_1(self, weights, bias, x):
        out_1 = self._perceptions(weights, bias, x)
        return out_1
