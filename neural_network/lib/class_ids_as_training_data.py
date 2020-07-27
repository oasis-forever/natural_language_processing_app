import numpy as np
from keras.utils import to_categorical
import sys
sys.path.append("../lib/concern")
from n_dim_generator import generate_n_dim
from mlp_builder import build_multi_layered_perceptron

class ClassIdsAsTrainingData:
    def __init__(self):
        pass

    def convert_to_categorical(self):
        y = np.array([0, 1, 2, 3, 4])
        y_one_hot = to_categorical(y)
        return y_one_hot

    def build_mlp(self):
        self.mlp = build_multi_layered_perceptron(loss="sparse_categorical_crossentropy")

    def fit_mlp(self):
        X = np.array([
            generate_n_dim(100),
            generate_n_dim(100),
        ])
        y = np.array([0, 1])
        self.mlp.fit(X, y, epochs=100)
