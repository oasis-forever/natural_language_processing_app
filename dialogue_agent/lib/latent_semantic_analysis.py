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

    def bow_table_to_barchart(self):
        draw_barcharts(self.bow.toarray(), self.vectorizer.get_feature_names(), self.texts, "bow_table")

    def svd_array_to_barchart(self):
        draw_barcharts(self.decomposed_feature, range(self.svd.n_components), self.texts, "svd_array")

if __name__ == "__main__":
    lsa = LatentSemanticAnalysis()
    texts = [
        "車は速く走る",
        "バイクは速く走る",
        "自転車はゆっくり走る",
        "三輪車はゆっくり走る",
        "プログラミングは楽しい",
        "Pythonは楽しい",
    ]
    lsa.vectorize(texts)
    lsa.execute_svd()
    lsa.bow_table_to_barchart()
    lsa.svd_array_to_barchart()
