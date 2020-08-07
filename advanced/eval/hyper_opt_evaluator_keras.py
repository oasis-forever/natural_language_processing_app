from os.path import dirname, join, normpath
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from hyper_opt_keras import HyperOptKeras
from data_preparation import prepare_data

if __name__ == "__main__":
    hoek = HyperOptKeras()
    hoek.feature_extraction()
    hoek.search_best_params()
    vectorizer, mlp = hoek.build_model_with_best_params()
    test_texts, test_labels = prepare_data("../csv/test_data.csv")
    test_vectors = vectorizer.transform(test_texts)
    test_preds = np.argmax(mlp.predict(test_vectors), axis=1)
    print(accuracy_score(test_preds, test_labels))
