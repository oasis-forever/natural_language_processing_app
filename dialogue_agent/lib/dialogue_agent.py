from os.path import dirname, join, normpath
import MeCab
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import neologdn
import unicodedata
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize

BASE_DIR = normpath(dirname("__file__"))

class DialogueAgent:
    def __init__(self, training_data):
        self.tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    def extract_trainig_data(self, training_data):
        training_data = pd.read_csv(join(BASE_DIR, training_data))
        self.texts = training_data["text"]
        self.labels = training_data["label"]

    def train(self, ngram_range):
        # Unify vectorizer and classifier into pipeline
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(tokenizer=lemmatize, ngram_range=ngram_range)),
            ("classifier", SVC())
        ])
        # Call vectorizer.fit(), vectorizer.transform() and classifier.fit() via pipeline.fit()
        pipeline.fit(self.texts, self.labels)
        # Sustain as an instance variable
        self.pipeline = pipeline

    def reply(self, input_text, replies):
        with open(join(BASE_DIR, replies)) as f:
            replies = f.read().split("\n")
        # Call vectorizer.transform() and classifier.predict() via pipeline.predict()
        predictions = self.pipeline.predict([input_text])
        # Assign the first element of list of class ids
        predicted_class_id = predictions[0]
        print(replies[predicted_class_id])
