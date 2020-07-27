from os.path import dirname, join, normpath
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from dialogue_agent import DialogueAgent

if __name__ == "__main__":
    BASE_DIR = normpath(dirname("__file__"))
    dialogue_agent = DialogueAgent()
    dialogue_agent.extract_trainig_data("../csv/training_data.csv")
    # Training
    dialogue_agent.train((1, 2))
    # Evaluation
    # Load test data
    test_data = pd.read_csv(join(BASE_DIR, "../csv/test_data.csv"))
    # Predict the whole test data all at once as list
    predictions = dialogue_agent.predict(test_data["text"])
    # Evaluate accuracy by comparing predictions with accurate class ids with sklearn.metrics.accuracy_score
    print(accuracy_score(test_data["label"], predictions))