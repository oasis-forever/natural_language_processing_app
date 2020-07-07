from os.path import dirname, join, normpath
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenize

class ExtractBowFromTrainingData:
    def __init__(self):
        pass

    def load_csv(self, csv):
        BASE_DIR = normpath(dirname("__file__"))
        csv_path = join(BASE_DIR, csv)
        training_data = pd.read_csv(csv_path)
        self.training_texts = training_data["text"]
        return self.training_texts

    def calc_bow(self):
        vectorizer = CountVectorizer(tokenizer=tokenize)
        # TODO: Solve a warning below
        # /home/oasist/.pyenv/versions/3.8.1/lib/python3.8/site-packages/sklearn/feature_extraction/text.py:484: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
        # warnings.warn("The parameter 'token_pattern' will not be used"
        vectorizer.fit(self.training_texts)
        bow = vectorizer.transform(self.training_texts)
        return bow
