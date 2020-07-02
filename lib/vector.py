import numpy as np

class Vector:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def sum(self, array):
        return self.array + array

    def multiply(self, array):
        return self.array * array

    def dot_product(self, array):
        return np.dot(self.array, array)
