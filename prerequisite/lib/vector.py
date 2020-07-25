import numpy as np

class Vector:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.max = np.max(array)
        self.min = np.min(array)
        self.argmax = np.argmax(array)
        self.argmin = np.argmin(array)

    def sum(self, array):
        return self.array + array

    def multiply(self, array):
        return self.array * array

    def dot_product(self, array):
        return np.dot(self.array, array)

    def v_stack(self, array):
        np.vstack((self.array, array))

    def h_stack(self, array):
        np.hstack((self.array, array))
