from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("./concern")
from tokenizer import tokenize

class SkLearnNgramTfIdf():
    def __init__(self):
        pass

    def vectorize(self, texts, ngram_range):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=ngram_range)
        self.vectorizer.fit(texts)
        self.bow = self.vectorizer.transform(texts)

    def bow_array(self):
        return self.bow.toarray()

    def vocabulary(self):
        return self.vectorizer.vocabulary_
