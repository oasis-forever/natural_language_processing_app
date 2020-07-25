import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import sys
sys.path.append("../lib/concern")
from n_dim_generator import generate_n_dim

class ClassIdsAsTrainingData:
    def __init__(self):
        pass

    def convert_to_categorical(self):
        y = np.array([0, 1, 2, 3, 4])
        y_one_hot = to_categorical(y)
        return y_one_hot

    def design_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=32, input_dim=100, activation="relu"))
        self.model.add(Dense(units=10, activation="softmax"))
        # sparce_categorical_crossentropy receives non-one-hot encode as training data
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    def fit_model(self):
        X = np.array([
            generate_n_dim(100),
            generate_n_dim(100),
        ])
        y = np.array([0, 1])
        self.model.fit(X, y, epochs=100)
