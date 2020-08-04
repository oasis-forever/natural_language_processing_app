from os.path import dirname, join, normpath
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../csv")
sys.path.append("./concern")
from lemmatizer import lemmatize

class GridSearch:
    def __init__(self):
        pass

    def _load_data(self, csv_data):
        BASE_DIR = normpath(dirname("__file__"))
        data = pd.read_csv(join(BASE_DIR, csv_data))
        return data

    def _get_texts(self, csv_data):
        data = self._load_data(csv_data)
        texts = data["text"]
        return texts

    def _get_labels(self, csv_data):
        data = self._load_data(csv_data)
        labels = data["label"]
        return labels

    def feature_extraction(self):
        train_texts = self._get_texts("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(train_texts)

    def search_best_params(self):
        train_labels = self._get_labels("../csv/training_data.csv")
        params = {
            "n_estimators": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
            "max_features": ("sqrt", "log2", None)
        }
        classifier = RandomForestClassifier()
        self.gridsearcher = GridSearchCV(classifier, params)
        self.gridsearcher.fit(self.train_vectors, train_labels)
        return self.gridsearcher.best_params_

    def classify_with_best_params(self):
        test_texts = self._get_texts("../csv/test_data.csv")
        test_vectors = self.vectorizer.transform(test_texts)
        predictions = self.gridsearcher.predict(test_vectors)
        return predictions
