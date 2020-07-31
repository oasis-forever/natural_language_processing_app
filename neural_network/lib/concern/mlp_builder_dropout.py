from keras.layers import Dense, Dropout
from keras.models import Sequential

def dropout_layered_perceptron(hidden_units=64, input_dim=100, i_activator="relu", output_dim=10, o_activator="softmax", loss="categorical_crossentropy", optimizer="adam"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation=i_activator))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=hidden_units, activation=i_activator))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=32, activation=i_activator))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp
