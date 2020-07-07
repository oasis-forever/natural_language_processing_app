from os.path import dirname, join, normpath
import MeCab
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

BASE_DIR = normpath(dirname("__file__"))

class DialogueAgent:
    def __init__(self, training_data):
        self.tagger = MeCab.Tagger()
        self.texts, self.labels = self._extract_trainig_data(training_data)

    def _extract_trainig_data(self, training_data):
        training_data = pd.read_csv(join(BASE_DIR, training_data))
        texts = training_data["text"]
        labels = training_data["label"]
        return texts, labels

    def _tokenize(self, text):
        node = self.tagger.parseToNode(text)
        tokens = []
        while node:
            if node.surface != "":
                tokens.append(node.surface)
            node = node.next
        return tokens

    def train(self):
        # Unify vectorizer and classifier into pipeline
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(tokenizer=self._tokenize)),
            ("classifier", SVC())
        ])
        # Call vectorizer.fit(), vectorizer.transform() and classifier.fit() via pipeline.fit()
        pipeline.fit(self.texts, self.labels)
        # Sustain as an instance variable
        self.pipeline = pipeline

    def predict(self, texts):
        # Call vectorizer.transform() and classifier.predict() via pipeline.predict()
        return self.pipeline.predict(texts)

    def reply(self, input_text, replies):
        self.train()
        with open(join(BASE_DIR, replies)) as f:
            replies = f.read().split("\n")
        input_text = input_text
        # predict method receives input_text as list
        predictions = self.predict([input_text])
        # Assign the first element of list of class ids
        predicted_class_id = predictions[0]
        print(replies[predicted_class_id])
