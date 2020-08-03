import numpy as np
from gensim.models import Word2Vec
from keras.layers import Embedding
import sys
sys.path.append("../ja")

class EmbeddingLayeredCnn:
    def __init__(self):
        self.we_model = Word2Vec.load("../ja/word2vec.gensim.model")

    def tokens_to_sequence(self, we_model, tokens):
        sequence = []
        for token in tokens:
            try:
                sequence.append(we_model.wv.vocab[token].index + 1)
            except KeyError:
                pass
        return sequence

    def get_keras_embedding(self, keyed_vectors, *args, **kwargs):
        weights = keyed_vectors.vectors
        word_num = weights.shape[0]
        embedding_dim = weights.shape[1]
        zero_word_vector = np.zeros((1, weights.shape[1]))
        weights_with_zero = np.vstack((zero_word_vector, weights))
        embedding_dim = Embedding(input_dim=word_num + 1, output_dim=embedding_dim, weights=[weights_with_zero], *args, **kwargs)
        return embedding_dim
