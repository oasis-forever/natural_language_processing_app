from os.path import dirname, join, normpath
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from dialogue_agent import DialogueAgent
from data_preparation import prepare_data

if __name__ == "__main__":
    BASE_DIR = normpath(dirname("__file__"))
    dialogue_agent = DialogueAgent()
    # Training
    dialogue_agent.train()
    # Evaluation
    test_texts, test_labels = prepare_data("../csv/test_data.csv")
    # Predict the whole test data all at once as list
    predictions = dialogue_agent.predict(test_texts)
    # Evaluate accuracy by comparing predictions with accurate class ids with sklearn.metrics.accuracy_score
    print(accuracy_score(test_labels, predictions))
