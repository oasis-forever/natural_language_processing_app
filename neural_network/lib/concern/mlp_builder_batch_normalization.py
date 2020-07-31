from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def batch_normalization(hidden_units=64, input_dim=100, i_activator="relu", output_dim=10, o_activator="softmax", loss="categorical_crossentropy", optimizer="adam"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=hidden_units, activation=i_activator))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=32, activation=i_activator))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp

def batch_normalization_before_activation(hidden_units=64, input_dim=100, i_activator="relu", output_dim=10, o_activator="softmax", loss="categorical_crossentropy", optimizer="adam"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim))
    mlp.add(BatchNormalization())
    mlp.add(Activation(i_activator))
    mlp.add(Dense(units=hidden_units))
    mlp.add(BatchNormalization())
    mlp.add(Activation(i_activator))
    mlp.add(Dense(units=32))
    mlp.add(BatchNormalization())
    mlp.add(Activation(i_activator))
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp
