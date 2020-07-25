import numpy as np
import random as r
from keras.layers import Dense
from keras.models import Sequential

class MultiClassRecongnizerPerceptron:
    def __init__(self):
        pass

    def _generate_n_dim(self, dim):
        p_float = []
        n_float = []
        for i in range(0, 11):
            p_float.append(float(i) / 10)
        for i in p_float:
            n_float.append(i * -1)
        p_float.sort(reverse=True)
        n_float.pop(0)
        float_list = p_float + n_float
        n_dim = r.choices(float_list, k=dim)
        return n_dim

    def design_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=32, input_dim=100, activation="relu"))
        self.model.add(Dense(units=10, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")

    def fit_model(self):
        X = np.array([
            self._generate_n_dim(100),
            self._generate_n_dim(100),
        ])
        y = np.array([
            # Class ID is 0
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Class ID is 1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        self.model.fit(X, y, epochs=100)
