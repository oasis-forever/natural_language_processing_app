import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers.wrappers import Bidirectional
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
sys.path.append("../csv")
sys.path.append("../ja")
sys.path.append("../lib")
sys.path.append("../lib/concern")
from embedding_layer import EmbeddingLayer
from lemmatizer import lemmatize

MAX_SEQUENCE_LENGTH = 20

class EmbeddingLayeredRnnEvaluator:
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
        self.model.add(self.embedding_layer.get_keras_embedding(self.embedding_layer.we_model.wv, input_shape=(MAX_SEQUENCE_LENGTH, ), mask_zero=True, trainable=False))

    def long_short_term_memory(self):
        self.model.add(Bidirectional(GRU(units=256), "concat"))
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
    elrc = EmbeddingLayeredRnnEvaluator()
    x_train, y_train = elrc.prepare_data("../csv/training_data.csv")
    elrc.build_model()
    elrc.long_short_term_memory()
    elrc.train(x_train, y_train)
    x_test, y_test = elrc.prepare_data("../csv/test_data.csv")
    elrc.predict(x_test)
    elrc.evaluate(y_test)
