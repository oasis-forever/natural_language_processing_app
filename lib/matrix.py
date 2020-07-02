import numpy as np

class Matrix:
    def __init__(self, list):
        self.array = list
        self.shape = list.shape

    def sum(self, num):
        return self.array + num

    def multiply(self, num):
        return self.array * num
