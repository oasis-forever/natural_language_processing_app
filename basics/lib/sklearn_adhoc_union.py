import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

# Matcher of regex for detection of the end of sentence
rx_periods = re.compile(r"[.。. ]+")

# Treat the whole length of sentence and the number of phrases splited by punctuation marks as feature, whose class implementing feature extraction is TextStats
# Inherit BaseEstimator and TransformerMixin Scikit-learn API provides
class TextStats(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    # Just return instance(nothing to learn)
    def fit(self, x, y=None):
        return self

    # Return lists of dict including the whole length of sentence and the number of phrases splited by punctuation marks
    def tranform(self, texts):
        return [
            {
                "length": len(text),
                "num_sentenses": len([sent for sent in rx_periods.split(text) if len(sent) > 0])
            }
            for text in texts
        ]

    def unite_feature(self, texts, ngram_range):
        # Unite lists of dict as a feature vector
        combined = FeatureUnion([
            # FIXME: https://github.com/oasis-forever/nlp_tutorial/issues/1
            ("stats", Pipeline([("stats", TextStats()), ("vect", DictVectorizer())])),
            ("char_bigram", CountVectorizer(analyzer="char", ngram_range=ngram_range))
        ])
        combined.fit(texts)
        feature = combined.tranform(texts)
        return feature.toarray()
