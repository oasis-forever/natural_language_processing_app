from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append("./concerns")
from tokenizer import tokenize

class SkLearnNgram():
    def __init__(self):
        pass

    def vectorize(self, texts, ngram_range):
        self.vectorizer = CountVectorizer(tokenizer=tokenize, ngram_range=ngram_range)
        self.vectorizer.fit(texts)
        self.bow = self.vectorizer.transform(texts)

    def bow_array(self):
        return self.bow.toarray()

    def vocabulary(self):
        return self.vectorizer.vocabulary_
