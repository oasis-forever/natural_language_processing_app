from keras.layers import Dense
from keras.models import Sequential
import sys
sys.path.append("../lib/concern")
from n_dim_generator import generate_n_dim

class MultiLayeredPerceptronKeras:
    def __init__(self):
        pass

    def build_mlp(self):
        self.mlp = Sequential()
        # 2 units, 3 dimensions
        self.mlp.add(Dense(units=2, activation="relu", input_dim=3))
        # inherit units=2 as input_dim(2 dimensions)
        self.mlp.add(Dense(units=1, activation="sigmoid"))
        self.mlp.compile(loss="binary_crossentropy", optimizer="adam")

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
