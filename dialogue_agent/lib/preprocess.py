from tokenizer import tokenize
import neologdn
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer

class PreProcess:
    def __init__(self):
        pass

    def _unicodedata_normalized_token(self, text):
        unicodedata_normalized_text = unicodedata.normalize("NFKC", text)
        return tokenize(unicodedata_normalized_text)

    def raw_tokenize(self, text):
        return tokenize(text)

    def neologdn_normalized_token(self, text):
        neologdn_normalized_text = neologdn.normalize(text)
        return tokenize(neologdn_normalized_text)

