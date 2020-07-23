from os.path import dirname, join, normpath
import MeCab
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import neologdn
import unicodedata
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
        # Unify vectorizer and classifier into pipeline
        self.pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(tokenizer=lemmatize, ngram_range=ngram_range)),
            ("classifier", RandomForestClassifier(n_estimators=30))
        ])
        # Call vectorizer.fit(), vectorizer.transform() and classifier.fit() via pipeline.fit()
        self.pipeline.fit(self.texts, self.labels)

    def predict(self, input_text):
        # Call vectorizer.transform() and classifier.predict() via pipeline.predict()
        self.predictions = self.pipeline.predict(input_text)
        return self.predictions

    def reply(self, replies):
        with open(join(BASE_DIR, replies)) as f:
            replies = f.read().split("\n")
        # Assign the first element of list of class ids
        predicted_class_id = self.predictions[0]
        return replies[predicted_class_id]
