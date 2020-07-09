from tokenizer import tokenize
import neologdn
import unicodedata

class Text:
    def __init__(self, text):
        self.text = text

    def raw_tokenize(self):
        return tokenize(self.text)

    def neologdn_normalized_token(self):
        neologdn_normalized_text = neologdn.normalize(self.text)
        return tokenize(neologdn_normalized_text)

    def unicodedata_normalized_text(self, text):
        unicodedata_normalized_text = unicodedata.normalize("NFKC", self.text)
        return unicodedata_normalized_text
