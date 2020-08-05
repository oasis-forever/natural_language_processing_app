import unicodedata
import os
from os.path import dirname, join, normpath
import MeCab
import neologdn
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from mlp_builder import double_mlp_relu
from data_preparation import prepare_data

class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger(os.environ['MECAB_IPADIC_NEOLOGD'])

    def train(self, ngram_range):
        training_texts, training_labels = prepare_data("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=ngram_range)
        self.vectorizer.fit(training_texts)
        # Dimension of feature vector to input
        feature_dim = len(self.vectorizer.get_feature_names())
        # Dimensions to output which is equal to the nunber of labels
        n_labels = max(training_labels) + 1
        classifier = KerasClassifier(build_fn=double_mlp_relu, input_dim=feature_dim, hidden_units=32, output_dim=n_labels)
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", classifier)
        ])
        # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/2#issuecomment-664194089
        self.pipeline.fit(training_texts, training_labels, classifier__epochs=100)

    def predict(self, input_text):
        self.predictions = self.pipeline.predict(input_text)

    def reply(self):
        BASE_DIR = normpath(dirname("__file__"))
        with open(join(BASE_DIR, "../csv/replies.csv")) as f:
            replies = f.read().split("\n")
        predicted_class_id = self.predictions[0]
        return replies[predicted_class_id]
