import os
from os.path import dirname, join, normpath
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import neologdn
import unicodedata
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from data_preparation import prepare_data

class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger(os.environ['MECAB_IPADIC_NEOLOGD'])

    def train(self):
        training_texts, training_labels = prepare_data("../csv/training_data.csv")
        # Unify vectorizer and classifier into pipeline
        self.pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))),
            ("classifier", RandomForestClassifier(n_estimators=30))
        ])
        # Call vectorizer.fit(), vectorizer.transform() and classifier.fit() via pipeline.fit()
        self.pipeline.fit(training_texts, training_labels)

    def predict(self, input_text):
        # Call vectorizer.transform() and classifier.predict() via pipeline.predict()
        self.predictions = self.pipeline.predict(input_text)
        return self.predictions

    def reply(self):
        BASE_DIR = normpath(dirname("__file__"))
        with open(join(BASE_DIR, "../csv/replies.csv")) as f:
            replies = f.read().split("\n")
        # Assign the first element of list of class ids
        predicted_class_id = self.predictions[0]
        return replies[predicted_class_id]
