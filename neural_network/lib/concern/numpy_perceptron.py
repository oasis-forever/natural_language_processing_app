import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def rectified_liner_unit(z):
    return np.maximum(z, 0)

def calc_innner_product(weights, bias, x):
    z = bias + np.dot(weights, x)
    return z
