from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append("./concerns")
from tokenizer import tokenize

class SkLearnBasic():
    def __init__(self):
        pass

    def calc_bow(self, texts):
        vectorizer = CountVectorizer(tokenizer=tokenize)
        vectorizer.fit(texts)
        bow = vectorizer.transform(texts)
        return bow.toarray()
