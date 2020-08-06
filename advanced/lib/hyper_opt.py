from hyperopt import fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from data_preparation import prepare_data

class HyperOpt:
    def __init__(self):
        pass

    def _objective(self, args):
        classifier = RandomForestClassifier(n_estimators=int(args["n_estimators"]), max_features=args["max_features"])
        classifier.fit(self.tr_vectors, self.tr_labels)
        val_predictions = classifier.predict(self.val_vectors)
        accuracy = accuracy_score(val_predictions, self.val_labels)
        return -accuracy

    def feature_extraction(self):
        training_texts, self.training_labels = prepare_data("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(training_texts)
        self.tr_labels, self.val_labels, self.tr_vectors, self.val_vectors = train_test_split(self.training_labels, self.train_vectors, random_state=42)

    def search_best_params(self):
        self.max_features_choices = ("sqrt", "log2", None)
        self.space = {
            # min: 10, max: 500, each: 10
            "n_estimators": hp.quniform("n_estimators", 10, 500, 10),
            "max_features": hp.choice("max_features", self.max_features_choices)
        }
        self.best = fmin(self._objective, self.space, algo=tpe.suggest, max_evals=30)

    def build_classifier_with_best_params(self):
        self.best_classifier = RandomForestClassifier(n_estimators=int(self.best["n_estimators"]), max_features=self.best["max_features"])
        self.best_classifier.fit(self.train_vectors, self.training_labels)
        return self.vectorizer, self.best_classifier
