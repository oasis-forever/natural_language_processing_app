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
            # FIXME: The following error is notified in executing test
            # Traceback (most recent call last):
            #  File "<stdin>", line 1, in <module>
            #  File "../lib/sklearn_adhoc_union.py", line 29, in unite_feature
            #    ("stats", Pipeline([("stats", TextStats()), ("vect", DictVectorizer())])),
            #  File "/home/oasist/.pyenv/versions/3.8.1/lib/python3.8/site-packages/sklearn/utils/#validation.py", line 73, in inner_f
            #    return f(**kwargs)
            #  File "/home/oasist/.pyenv/versions/3.8.1/lib/python3.8/site-packages/sklearn/#pipeline.py", line 114, in __init__
            #    self._validate_steps()
            #  File "/home/oasist/.pyenv/versions/3.8.1/lib/python3.8/site-packages/sklearn/#pipeline.py", line 159, in _validate_steps
            #    raise TypeError("All intermediate steps should be "
            #TypeError: All intermediate steps should be transformers and implement fit and transform #or be the string 'passthrough' 'TextStats()' (type <class #'sklearn_adhoc_union.TextStats'>) doesn't
            ("stats", Pipeline([("stats", TextStats()), ("vect", DictVectorizer())])),
            ("char_bigram", CountVectorizer(analyzer="char", ngram_range=ngram_range))
        ])
        combined.fit(texts)
        feature = combined.tranform(texts)
        return feature.toarray()
