from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
sys.path.append("../csv")
sys.path.append("../ja")
sys.path.append("../lib")
sys.path.append("../lib/concern")
from simple_we_classification import SimpleWeClassification
import numpy as np
import pandas as pd

class SimpleWeClassificationEvaluator:
    def __init__(self):
        self.swc = SimpleWeClassification()

    def prepare_data(self, csv_data):
        date = pd.read_csv(csv_data)
        X_data = np.array([self.swc.calc_text_feature(text) for text in date["text"]])
        y_data = np.array(date["label"])
        return X_data, y_data

    def train(self, X_train, y_train):
        self.svc = SVC()
        self.svc.fit(X_train, y_train)

    def predict(self, X_test):
        self.y_pred = self.svc.predict(X_test)

    def evaluate(self, y_test):
        print(accuracy_score(y_test, self.y_pred))

if __name__ == "__main__":
    swce = SimpleWeClassificationEvaluator()
    X_train, y_train = swce.prepare_data("../csv/training_data.csv")
    swce.train(X_train, y_train)
    X_test, y_test = swce.prepare_data("../csv/test_data.csv")
    swce.predict(X_test)
    swce.evaluate(y_test)
