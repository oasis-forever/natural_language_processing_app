from keras.layers import Dense
from keras.models import Sequential

def build_multi_layered_perceptron(hidden_units=32, input_dim=100, output_dim=10, loss="categorical_crossentropy"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation="relu"))
    mlp.add(Dense(units=output_dim, activation="softmax"))
    mlp.compile(loss=loss, optimizer="adam")
    return mlp
