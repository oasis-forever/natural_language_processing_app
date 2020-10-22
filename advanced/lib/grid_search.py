from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../csv")
sys.path.append("./concerns")
from lemmatizer import lemmatize
from data_preparation import texts_data, labels_data

class GridSearch:
    def __init__(self):
        pass

    def feature_extraction(self):
        training_texts = texts_data("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(training_texts)

    def search_best_params(self):
        training_labels = labels_data("../csv/training_data.csv")
        params = {
            "n_estimators": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
            "max_features": ("sqrt", "log2", None)
        }
        classifier = RandomForestClassifier()
        self.gridsearcher = GridSearchCV(classifier, params)
        self.gridsearcher.fit(self.train_vectors, training_labels)
        return self.gridsearcher.best_params_

    def classify_with_best_params(self):
        test_texts = texts_data("../csv/test_data.csv")
        test_vectors = self.vectorizer.transform(test_texts)
        predictions = self.gridsearcher.predict(test_vectors)
        return predictions
