from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append("./concern")
from tokenizer import tokenize
from data_preparation import texts_data

class ExtractBowFromTrainingData:
    def __init__(self):
        pass

    def calc_bow(self):
        training_texts = texts_data("../csv/training_data.csv")
        vectorizer = CountVectorizer(tokenizer=tokenize)
        # TODO: Solve a warning below
        # /home/oasist/.pyenv/versions/3.8.1/lib/python3.8/site-packages/sklearn/feature_extraction/text.py:484: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
        # warnings.warn("The parameter 'token_pattern' will not be used"
        vectorizer.fit(training_texts)
        bow = vectorizer.transform(training_texts)
        return bow
