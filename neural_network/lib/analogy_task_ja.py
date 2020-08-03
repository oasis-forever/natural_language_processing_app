from gensim.models import Word2Vec
import sys
sys.path.append("../ja")

class AnalogyTaskJa:
    def __init__(self):
        # Use gensim.downloader and download word embeddings model ready in advance
        # Unzip "../ja/ja.tsv.zip" before loading 
        self.model = Word2Vec.load("../ja/ja.bin")

    def word_embeddings(self):
        self.tokyo = self.model["東京"]
        self.japan = self.model["日本"]
        self.france = self.model["フランス"]

    # Get vector by means of calculation of each vector
    def calc_vector(self):
        # Japanese capital -> capital -> French capital
        self.vector = self.tokyo - self.japan + self.france

    def similar_words(self, vector):
        # FIXME: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
        return self.model.wv.similar_by_vector(vector)
