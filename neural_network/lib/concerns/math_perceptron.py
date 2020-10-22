import math

def rectified_liner_unit(z):
    return max(z, 0)

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))
