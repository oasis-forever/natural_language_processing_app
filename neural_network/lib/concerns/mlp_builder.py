from keras import regularizers, initializers
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def double_mlp_relu(hidden_units=32, input_dim=100, output_dim=10, o_activator="softmax", loss="categorical_crossentropy"):
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=output_dim, activation=o_activator))
    mlp.compile(loss=loss, optimizer="adam")
    return mlp

def quadruple_mlp_relu():
    mlp = Sequential()
    mlp.add(Dense(units=32, input_dim=100, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=32, input_dim=100, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=32, input_dim=100, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp

def mlp_relu_dropout():
    mlp = Sequential()
    mlp.add(Dense(units=64, input_dim=100, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=32, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp

def mlp_relu_batch_normalization():
    mlp = Sequential()
    mlp.add(Dense(units=64, input_dim=100, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=32, activation="relu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp

def mlp_relu_batch_normalization_before_activation():
    mlp = Sequential()
    mlp.add(Dense(units=64, input_dim=100, kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Activation("relu"))
    mlp.add(Dense(units=64, kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Activation("relu"))
    mlp.add(Dense(units=32, kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(BatchNormalization())
    mlp.add(Activation("relu"))
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp

def mlp_selu():
    mlp = Sequential()
    mlp.add(Dense(units=64, input_dim=100, activation="selu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=64, activation="selu", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp

def mlp_leakyrelu():
    mlp = Sequential()
    mlp.add(Dense(units=64, input_dim=100, activation=LeakyReLU(0.3), kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=64, activation=LeakyReLU(0.3), kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.add(Dense(units=10, activation="softmax", kernel_regularizer=regularizers.l2(0.1), kernel_initializer=initializers.glorot_normal()))
    mlp.compile(loss="categorical_crossentropy", optimizer="adam")
    return mlp
