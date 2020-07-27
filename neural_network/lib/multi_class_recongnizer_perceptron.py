import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys
sys.path.append("../lib/concern")
from n_dim_generator import generate_n_dim

class MultiClassRecongnizerPerceptron:
    def __init__(self):
        pass

    def build_mlp(self):
        self.mlp = Sequential()
        self.mlp.add(Dense(units=32, input_dim=100, activation="relu"))
        self.mlp.add(Dense(units=10, activation="softmax"))
        self.mlp.compile(loss="categorical_crossentropy", optimizer="adam")

    def fit_mlp(self):
        X = np.array([
            generate_n_dim(100),
            generate_n_dim(100),
        ])
        y = np.array([
            # Class ID is 0
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Class ID is 1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        self.mlp.fit(X, y, epochs=100)
