import numpy as np

class Vector:
    def __init__(self, list):
        self.array = list
        self.shape = list.shape

    def sum(self, list):
        return self.array + list

    def multiply(self, list):
        return self.array * list

