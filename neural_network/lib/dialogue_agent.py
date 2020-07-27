import unicodedata
from os.path import dirname, join, normpath
import MeCab
import neologdn
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize

BASE_DIR = normpath(dirname("__file__"))

class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    def extract_trainig_data(self, training_data):
        training_data = pd.read_csv(join(BASE_DIR, training_data))
        self.texts = training_data["text"]
        self.labels = training_data["label"]

    def train(self, ngram_range):
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=ngram_range)
        tfidf = self.vectorizer.fit_transform(self.texts)
        # Dimension of feature vector to input
        feature_dim = len(self.vectorizer.get_feature_names())
        # Dimensions to output which is equal to the nunber of labels
        n_labels = max(self.labels) + 1
        # Build multi-layered perceptron
        self.mlp = Sequential()
        self.mlp.add(Dense(units=32, input_dim=feature_dim, activation="relu"))
        self.mlp.add(Dense(units=n_labels, activation="softmax"))
        self.mlp.compile(loss="categorical_crossentropy", optimizer="adam")
        # Convert labels into one-hot encode to class IDs are used as traiing data
        labels_onehot = to_categorical(self.labels, n_labels)
        # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/2
        self.mlp.fit(tfidf, labels_onehot, epochs=100)

    def predict(self, input_text):
        tfidf = self.vectorizer.transform(input_text)
        predictions = self.mlp.predict(tfidf)
        # Treat the biggest value of index of dimension as predicted_class_id
        self.predicted_labels = np.argmax(predictions, axis=1)

    def reply(self, replies):
        with open(join(BASE_DIR, replies)) as f:
            replies = f.read().split("\n")
        predicted_class_id = self.prediction_labels[0]
        return replies[predicted_class_id]
