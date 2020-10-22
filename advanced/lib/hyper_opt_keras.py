import math
from hyperopt import fmin, hp, tpe
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("./concerns")
from lemmatizer import lemmatize
from data_preparation import prepare_data

class HyperOptKeras:
    def __init__(self):
        pass

    def _build_mlp_model(self, hidden_units, dropout, optimizer):
        mlp = Sequential()
        mlp.add(Dense(units=hidden_units, input_dim=self.input_dim, activation="relu"))
        if dropout:
            mlp.add(Dropout(dropout))
        mlp.add(Dense(units=self.output_dim, activation="softmax"))
        mlp.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
        return mlp

    def _objective(self, args):
        if K.backend() == "tensorflow":
            # Release memory to avoid capacity erosion by building each model
            K.clear_session()
        hidden_units = int(args["hidden_units"])
        dropout = args["dropout"]
        optimizer_class, optimizer_args = args["optimizer"]
        optimizer = optimizer_class(**optimizer_args)
        mlp = self._build_mlp_model(hidden_units, dropout, optimizer)
        batch_size = max(int(args["batch_size"]), 1)
        history = mlp.fit(
            self.tr_vectors,
            self.tr_labels,
            epochs=self.train_epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(min_delta=0.0, patience=3)],
            validation_data=(self.val_vectors, self.val_labels)
        )
        if len(history.history["val_loss"]) == self.train_epochs:
            print("[WARNING] Early stopping did not work")
        val_pred = np.argmax(mlp.predict(self.val_vectors), axis=1)
        accuracy = accuracy_score(val_pred, self.val_labels)
        return -accuracy

    def feature_extraction(self):
        training_texts, self.training_labels = prepare_data("../csv/training_data.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=lemmatize, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(training_texts)
        feature_dim = self.train_vectors.shape[1]
        n_labels = max(self.training_labels) + 1
        self.tr_labels, self.val_labels, self.tr_vectors, self.val_vectors = train_test_split(self.training_labels, self.train_vectors, random_state=42)
        self.input_dim = feature_dim
        self.output_dim = n_labels
        self.train_epochs = 200

    # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/5
    def search_best_params(self):
        self.space = {
            "optimizer": hp.choice("optimizer", [
                (SGD,
                    {
                        "lr": hp.loguniform("lr_sgd", math.log(1e-6), math.log(1)),
                        "momentum": hp.uniform("momentum", 0, 1)
                    }
                ),
                (Adagrad,
                    {
                        "lr": hp.loguniform("lr_adagrad", math.log(1e-6), math.log(1))
                    }
                ),
                (Adadelta,
                    {
                        "lr": hp.loguniform("lr_adadelta", math.log(1e-6), math.log(1))
                    }
                ),
                (Adam,
                    {
                        "lr": hp.loguniform("lr_adam", math.log(1e-6), math.log(1))
                    }
                )
            ]),
            "hidden_units": hp.qloguniform("hidden_units", math.log(32), math.log(256), 1),
            "batch_size": hp.qloguniform("batch_size", math.log(1), math.log(256), 1),
            "dropout": hp.uniform("dropout", 0, 0.5)
        }
        self.best = fmin(self._objective, self.space, algo=tpe.suggest, max_evals=100)

    def build_model_with_best_params(self):
        optimier_choices = [node.pos_args[0].obj for node in space["optimizer"].pos_args[1:]]
        BestOptimizer = optimier_choices[self.best["optimier"]]
        optimier_args = {}
        if BestOptimizer == SGD:
            optimier_args["lr"] = self.best["lr_sgd"]
            optimier_args["momentum"] = self.best['momentum']
        elif BestOptimizer == Adagrad:
            optimier_args["lr"] = self.best["lr_adagrad"]
        elif BestOptimizer == Adadelta:
            optimier_args["lr"] = self.best["lr_adadelta"]
        elif BestOptimizer == Adam:
            optimier_args["lr"] = self.best["lr_adam"]
        optimizer = BestOptimizer(**optimier_args)
        hidden_units = int(self.best["hidden_units"])
        dropout = self.best["dropout"]
        mlp = self._build_mlp_model(hidden_units, dropout, optimizer)
        batch_size = max(int(self.best["batch_size"]), 1)
        mlp.fit(
            self.train_vectors,
            self.training_labels,
            epochs=self.train_epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(min_delta=0.0, patience=1)]
        )
        return self.vectorizer, self.mlp
