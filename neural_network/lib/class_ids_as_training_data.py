import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import sys
sys.path.append("./concern")
from n_dim_generator import generate_n_dim
from mlp_builder import build_two_layered_perceptron

class ClassIdsAsTrainingData:
    def __init__(self):
        pass

    def convert_to_categorical(self):
        y = np.array([0, 1, 2, 3, 4])
        y_one_hot = to_categorical(y)
        return y_one_hot

    def build_mlp(self):
        self.mlp = build_two_layered_perceptron(loss="sparse_categorical_crossentropy")

    def fit_mlp(self):
        X = np.array([
            generate_n_dim(100),
            generate_n_dim(100),
        ])
        y = np.array([0, 1])
        # min_delta is a criterion judged as the lowest value of performance improvement
        # patience conditions a vakue where improvement has to be realised.  Otherwise, training will stop.
        self.mlp.fit(X, y, epochs=100, validation_split=0.1, callbacks=[EarlyStopping(min_delta=0.0, patience=1)])
