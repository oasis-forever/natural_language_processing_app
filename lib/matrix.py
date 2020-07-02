import numpy as np

class Matrix:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def sum(self, num):
        return self.array + num

    def multiply(self, num):
        return self.array * num

    def dot_product(self, array):
        return np.dot(self.array, array)
