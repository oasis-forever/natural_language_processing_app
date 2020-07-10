from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("./concern")
from tokenizer import tokenize

class SkLearnTfIdf():
    def __init__(self):
        pass

    def calc_tfidf(self, texts):
        vectorizer = TfidfVectorizer(tokenizer=tokenize)
        vectorizer.fit(texts)
        tfidf = vectorizer.transform(texts)
        return tfidf.toarray()
