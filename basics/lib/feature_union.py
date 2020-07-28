import scipy
from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append("./concern")
from tokenizer import tokenize

class FeatureUnion:
    def __init__(self, texts, ngram_range):
        # Word-based BoW
        self.word_bow_vectorizer = CountVectorizer(tokenizer=tokenize)
        self.word_bow_vectorizer.fit(texts)
        self.word_bow = self.word_bow_vectorizer.transform(texts)
        # bi-gram BoW
        self.char_bigram_vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range)
        self.char_bigram_vectorizer.fit(texts)
        self.char_bigram_bow = self.char_bigram_vectorizer.transform(texts)

    def word_bow_array(self):
        return self.word_bow.toarray()

    def char_bigram_bow_array(self):
        return self.char_bigram_bow.toarray()

    def unite_feature(self):
        feature = scipy.sparse.hstack((self.word_bow, self.char_bigram_bow))
        return feature.toarray()
