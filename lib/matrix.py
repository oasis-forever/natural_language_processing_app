import numpy as np

class Matrix:
    def __init__(self, list):
        self.array = list
        self.shape = list.shape

    def sum(self, num):
        return self.array + num

    def multiply(self, num):
        return self.array * num

    def dot_product(self, list):
        return np.dot(self.array, list)
