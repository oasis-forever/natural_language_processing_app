from keras.layers import Dense
from keras.models import Sequential

def build_double_layered_perceptron(hidden_units=32, input_dim=100, i_activator="relu", output_dim=10, o_activator="softmax", loss="categorical_crossentropy", optimizer="adam"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp

def build_quadruple_layered_perceptron(hidden_units=32, input_dim=100, i_activation="relu", output_dim=10, o_activation="softmax", loss="categorical_crossentropy", optimizer="adam"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp
