from keras.layers import Dense
from keras.models import Sequential

class MultiLayeredPerceptronKeras:
    def __init__(self):
        pass

    def design_model(self):
        self.model = Sequential()
        # 2 units, 3 dimensions
        self.model.add(Dense(units=2, activation="relu", input_dim=3))
        # inherit units=2 as input_dim(2 dimensions)
        self.model.add(Dense(units=1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam")

    def get_model_weights(self):
        return self.model.layers[0].get_weights()
