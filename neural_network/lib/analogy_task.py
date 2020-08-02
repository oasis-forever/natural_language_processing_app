import gensim.downloader as api

class AnalogyTask:
    def __init__(self):
        # Use gensim.downloader and download word embeddings model ready in advance
        self.model = api.load("glove-wiki-gigaword-50")

    def word_embeddings(self):
        self.tokyo = self.model["tokyo"]
        self.japan = self.model["japan"]
        self.france = self.model["france"]

    # Get vector by means of calculation of each vector
    def calc_vector(self):
        # Japanese capital -> capital -> French capital
        self.vector = self.tokyo - self.japan + self.france

    def closest_word(self):
        # FIXME: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
        return self.model.wv.similar_by_vector(self.vector, topn=1)[0]
