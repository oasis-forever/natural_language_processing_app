from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
import sys
sys.path.append("./concern")
from tokenizer import tokenize

class SkLearnFeatureUnion:
    def __init__(self, ngram_range):
        # Word-based BoW
        self.word_bow_vectorizer = CountVectorizer(tokenizer=tokenize)
        self.char_bigram_vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range)
        # Unite plural vectorizers to execute fit and transform at once
        self.estimators = [
            ("bow", self.word_bow_vectorizer),
            ("char_bigram", self.char_bigram_vectorizer)
        ]

    def unite_feature(self, texts):
        combined = FeatureUnion(self.estimators)
        combined.fit(texts)
        feature = combined.transform(texts)
        return feature.toarray()
