import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import sys
sys.path.append("./concern")
from n_dim_generator import generate_n_dim
from mlp_builder import double_mlp_relu

class MultiClassRecongnizerPerceptron:
    def __init__(self):
        pass

    def build_mlp(self):
        self.mlp = double_mlp_relu()

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
        # min_delta is a criterion judged as the lowest value of performance improvement
        # patience conditions a vakue where improvement has to be realised.  Otherwise, training will stop.
        self.mlp.fit(X, y, epochs=100, validation_split=0.1, callbacks=[EarlyStopping(min_delta=0.0, patience=1)])
