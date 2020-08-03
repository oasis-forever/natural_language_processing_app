import numpy as np
from gensim.models import Word2Vec
import sys
sys.path.append("./concern")
sys.path.append("../ja")
from lemmatizer import lemmatize

class SimpleWeClassification:
    def __init__(self):
        self.model = Word2Vec.load("../ja/word2vec.gensim.model")

    def calc_text_feature(self, text):
        """
        Calculate the feature of the text based on distributed representation
        Tokenize the text and get the distributed representation of each token
        Add up all distributed representations as the feautre of the text
        """
        tokens = lemmatize(text)
        word_vectors = np.empty((0, self.model.wv.vector_size))
        for token in tokens:
            try:
                word_vector = self.model[token]
                word_vectors = np.vstack((word_vectors, word_vector))
            except KeyError:
                pass
        if word_vectors.shape[0] == 0:
            return np.zeros(self.model.wv.vector_size)
        return np.sum(word_vectors, axis=0)
