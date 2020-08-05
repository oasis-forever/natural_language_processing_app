from os.path import dirname, join, normpath
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from hyper_opt import HyperOpt
from data_preparation import prepare_data

if __name__ == "__main__":
    hoe = HyperOpt()
    hoe.feature_extraction()
    hoe.search_best_params()
    vectorizer, best_classifier = hoe.build_classifier_with_best_params()
    test_texts, test_labels = prepare_data("../csv/test_data.csv")
    test_vectors = vectorizer.transform(test_texts)
    predictions = best_classifier.predict(test_vectors)
    print(accuracy_score(test_labels, predictions))
