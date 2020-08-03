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

if __name__ == "__main__":
    swc = SimpleWeClassification()
    training_date = pd.read_csv("../csv/training_data.csv")
    test_date = pd.read_csv("../csv/test_data.csv")

    X_train = np.array([swc.calc_text_feature(text) for text in training_date["text"]])
    y_train = np.array(training_date["label"])

    X_test = np.array([swc.calc_text_feature(text) for text in test_date["text"]])
    y_test = np.array(test_date["label"])

    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(accuracy_score(y_test, y_pred))
