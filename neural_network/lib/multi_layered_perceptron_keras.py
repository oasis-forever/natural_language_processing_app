import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys
sys.path.append("./concern")
from n_dim_generator import generate_n_dim
from mlp_builder import build_two_layered_perceptron

class MultiLayeredPerceptronKeras:
    def __init__(self):
        pass

    def build_mlp(self):
        self.mlp = build_two_layered_perceptron(hidden_units=2, input_dim=3, output_dim=1, o_activator="sigmoid", loss="binary_crossentropy")

    def fit_mlp(self):
        X = np.array([
            generate_n_dim(3),
            generate_n_dim(3),
        ])
        y = np.array([
            0,
            1
        ])
        self.mlp.fit(X, y, batch_size=32, epochs=100)
