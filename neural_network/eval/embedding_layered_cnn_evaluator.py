import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
sys.path.append("../csv")
sys.path.append("../ja")
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from embedding_layer import EmbeddingLayer
from lemmatizer import lemmatize

MAX_SEQUENCE_LENGTH = 20

class EmbeddingLayeredCnnEvaluator:
    def __init__(self):
        self.embedding_layer = EmbeddingLayer()

    def prepare_data(self, csv_data):
        # Load data
        data = pd.read_csv(csv_data)
        # Tokenize and make index
        texts = data["text"]
        lemmatized_texts = [lemmatize(text) for text in texts]
        sequences = [self.embedding_layer.tokens_to_sequence(self.embedding_layer.we_model, tokens) for tokens in lemmatized_texts]
        # The whole length of training_sequences corresponds with MAX_SEQUENCE_LENGTH
        x_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # Make class_id list as training_data
        y_data = np.asarray(data["label"])
        self.n_classes = max(y_data) + 1
        return x_data, y_data

    def build_model(self):
        self.model = Sequential()
        self.model.add(self.embedding_layer.get_keras_embedding(self.embedding_layer.we_model.wv, input_shape=(MAX_SEQUENCE_LENGTH, ), trainable=False))

    def one_d_convolution(self):
        self.model.add(Conv1D(filters=256, kernel_size=2, strides=1, activation="relu"))

    def global_max_pooling(self):
        self.model.add(MaxPooling1D(pool_size=int(self.model.output.shape[1])))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=self.n_classes, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def train(self, x_train, y_train):
        self.model.fit(x_train, to_categorical(y_train), epochs=50)

    def predict(self, x_test):
        self.y_pred = np.argmax(self.model.predict(x_test), axis=1)

    def evaluate(self, y_test):
        print(accuracy_score(y_test, self.y_pred))

if __name__ == "__main__":
    elce = EmbeddingLayeredCnnEvaluator()
    x_train, y_train = elce.prepare_data("../csv/training_data.csv")
    elce.build_model()
    elce.one_d_convolution()
    elce.global_max_pooling()
    elce.train(x_train, y_train)
    x_test, y_test = elce.prepare_data("../csv/test_data.csv")
    elce.predict(x_test)
    elce.evaluate(y_test)
