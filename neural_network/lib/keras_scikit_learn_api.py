import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import numpy as np
import sys
sys.path.append("./concern")
from n_dim_generator import generate_n_dim
from mlp_builder import build_two_layered_perceptron

class KerasScikitLearnApi:
    def __init__(self):
        pass

    def fit_classifier(self):
        X = np.array([
            generate_n_dim(100),
            generate_n_dim(100),
        ])
        y = np.array([0, 1])
        input_dim = X.shape[1]
        n_labels = max(y) + 1
        self.classifier = KerasClassifier(build_fn=build_two_layered_perceptron, input_dim=input_dim, hidden_units=32, output_dim=n_labels)
        self.classifier.fit(X, y, epochs=100)

    def predict(self, some_feature):
        return self.classifier.predict(some_feature)
