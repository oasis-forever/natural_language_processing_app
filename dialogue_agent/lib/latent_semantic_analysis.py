from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import sys
sys.path.append("./concern")
from lemmatizer import lemmatize
from barcharts_drawer import draw_barcharts
import pandas as pd
from matplotlib import pyplot as plt

class LatentSemanticAnalysis:
    def __init__(self):
        pass

    def _print_histogram(self, value_table, x_labels, titles, method_name):
        draw_barcharts(value_table, x_labels, titles, method_name)

    def vectorize(self, texts):
        self.texts = texts
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        self.vectorizer.fit(self.texts)
        self.bow = self.vectorizer.transform(self.texts)

    def bow_shape(self):
        print("Shape: {}".format(self.bow.shape))

    def bow_table(self):
        bow_table = pd.DataFrame(self.bow.toarray(), columns=self.vectorizer.get_feature_names())
        print(bow_table)

    def execute_svd(self):
        self.svd = TruncatedSVD(n_components=4, random_state=42)
        self.svd.fit(self.bow)
        self.decomposed_feature = self.svd.transform(self.bow)

    def svd_shape(self):
        print("Shape: {}".format(self.decomposed_feature.shape))

    def svd_array(self):
        return self.decomposed_feature

