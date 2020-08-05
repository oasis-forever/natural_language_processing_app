import os
from os.path import dirname, join, normpath
import MeCab
import pandas as pd
import neologdn
import unicodedata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from data_preparation import prepare_data

class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger(os.environ['MECAB_IPADIC_NEOLOGD'])

    def train(self):
        training_texts, training_labels = prepare_data("../csv/training_data.csv")
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(tokenizer=lemmatize)),
            ("classifier", RandomForestClassifier())
        ])
        params = {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            "class__n_estimators": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
            "class__max_features": ("sqrt", "log2", None)
        }
        self.clf = GridSearchCV(pipeline, params)
        self.clf.fit(training_texts, training_labels)

    def predict(self, input_text):
        return self.clf.predict(input_text)

    def reply(self):
        BASE_DIR = normpath(dirname("__file__"))
        with open(join(BASE_DIR, "../csv/replies.csv")) as f:
            replies = f.read().split("\n")
        predicted_class_id = self.predictions[0]
        return replies[predicted_class_id]
