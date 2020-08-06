from keras import backend as k
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../csv")
sys.path.append("./concern")
from lemmatizer import lemmatize
from data_preparation import texts_data, labels_data

class GridSearchKeras:
    def __init__(self):
        pass

    def _build_model(self, input_dim, output_dim, optimizer_class, learning_rate, dropout=0):
        if k.backend() == "tensorflow":
            # Release memory to avoid capacity erosion by building each model
            k.clear_session()
        mlp = Sequential()
        mlp.add(Dense(units=32, input_dim=input_dim, activation="relu"))
        if dropout:
            mlp.add(Dropout(dropout))
        mlp.add(Dense(units=output_dim, activation="softmax"))
        optimizer = optimizer_class(lr=learning_rate)
        mlp.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
        return mlp

    def feature_extraction(self):
        training_texts = texts_data("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(training_texts)

    def listup_params(self):
        self.training_labels = labels_data("../csv/training_data.csv")
        self.params = {
            "optimizer_class": [SGD, Adadelta, Adagrad, Adam],
            "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001],
            "dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "epochs": [10, 50, 100, 200],
            "batch_size": [16, 32, 64]
        }

    def search_best_params(self):
        feature_dim = self.train_vectors.shape[1]
        n_labels = max(self.training_labels) + 1
        model = KerasClassifier(build_fn=self._build_model, input_dim=feature_dim, output_dim=n_labels, verbose=0)
        self.gridsearcher = GridSearchCV(estimator=model, param_grid=self.params)
        self.gridsearcher.fit(self.train_vectors, self.training_labels)
        print(self.gridsearcher.best_params_)

    def classify_with_best_params(self):
        test_texts = texts_data("../csv/test_data.csv")
        test_vectors = self.vectorizer.transform(test_texts)
        predictions = self.gridsearcher.predict(test_vectors)
        return predictions
