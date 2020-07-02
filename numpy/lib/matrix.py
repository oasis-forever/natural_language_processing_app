import numpy as np

class Matrix:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.v_max = np.max(array, axis=0)
        self.h_max = np.max(array, axis=1)
        self.v_min = np.min(array, axis=0)
        self.h_min = np.min(array, axis=1)
        self.v_argmax = np.argmax(array, axis=0)
        self.h_argmax = np.argmax(array, axis=1)
        self.v_argmin = np.argmin(array, axis=0)
        self.h_argmin = np.argmin(array, axis=1)

    def sum(self, num):
        return self.array + num

    def multiply(self, num):
        return self.array * num

    def dot_product(self, array):
        return np.dot(self.array, array)

    def np_sum(self):
        return np.sum(self.array)

    def np_mean(self):
        return np.mean(self.array)
