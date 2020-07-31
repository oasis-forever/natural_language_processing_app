import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import sys
sys.path.append("./concern")
from n_dim_generator import generate_n_dim
from mlp_builder import build_double_layered_perceptron

class MultiLayeredPerceptronKeras:
    def __init__(self):
        pass

    def build_mlp(self):
        self.mlp = build_double_layered_perceptron(hidden_units=2, input_dim=3, output_dim=1, o_activator="sigmoid", loss="binary_crossentropy")

    def fit_mlp(self):
        X = np.array([
            generate_n_dim(3),
            generate_n_dim(3),
        ])
        y = np.array([
            0,
            1
        ])
        # min_delta is a criterion judged as the lowest value of performance improvement
        # patience conditions a vakue where improvement has to be realised.  Otherwise, training will stop.
        self.mlp.fit(X, y, batch_size=32, epochs=100, validation_split=0.1, callbacks=[EarlyStopping(min_delta=0.0, patience=1)])
