from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from barcharts_drawer import draw_barcharts
import pandas as pd
from matplotlib import pyplot as plt
import re

class LatentSemanticAnalysis:
    def __init__(self):
        pass

    def _filename(self, method_name):
        class_name = re.sub("([A-Z])", lambda x: "_" + x.group(0).lower(), str(self.__class__.__name__)).lstrip("_")
        filename = class_name + "_" + method_name
        return filename

    def vectorize(self, texts):
        self.texts = texts
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        self.vectorizer.fit(self.texts)
        self.bow = self.vectorizer.transform(self.texts)

    def bow_shape(self):
        return self.bow.shape

    def bow_table(self):
        bow_table = pd.DataFrame(self.bow.toarray(), columns=self.vectorizer.get_feature_names())
        draw_barcharts(self.bow.toarray(), self.vectorizer.get_feature_names(), self.texts, self._filename(sys._getframe().f_code.co_name))
        print(bow_table)

    def execute_svd(self):
        self.svd = TruncatedSVD(n_components=4, random_state=42)
        self.svd.fit(self.bow)
        self.decomposed_feature = self.svd.transform(self.bow)

    def svd_shape(self):
        return self.decomposed_feature.shape

    def svd_array(self):
        draw_barcharts(self.decomposed_feature, range(self.svd.n_components), self.texts, self._filename(sys._getframe().f_code.co_name))
        return self.decomposed_feature
