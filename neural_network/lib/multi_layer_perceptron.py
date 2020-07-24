class MultiLayerPerceptron:
    def __init__(self):
        pass

    def _rectified_liner_unit(self, z):
        return max(z, 0)

    def _perceptions(self, weights, biases, x):
        z_11 = biases[0] + x[0] * weights[0][0] + x[1] * weights[0][1] + x[2] * weights[0][2]
        output1 = self._rectified_liner_unit(z_11)
        z_12 = biases[1] + x[0] * weights[1][0] + x[1] * weights[1][1] + x[2] * weights[1][2]
        output2 = self._rectified_liner_unit(z_12)
        return [output1, output2]

    def layer_1(self, weights, biases, x):
        out_1 = self._perceptions(weights, biases, x)
        return out_1
