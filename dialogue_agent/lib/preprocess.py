import sys
sys.path.append("./concern")
from tokenizer import tokenize
from lemmatizer import lemmatize
import neologdn
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer

class PreProcess:
    def __init__(self):
        pass

    def _unicodedata_token(self, text):
        unicodedata_normalized_text = unicodedata.normalize("NFKC", text)
        return lemmatize(unicodedata_normalized_text)

    def _unicodedata_token(self, text):
        unicodedata_normalized_text = unicodedata.normalize("NFKC", text)
        return tokenize(unicodedata_normalized_text)

    def _unicodedata_normalized_token(self, text):
        unicodedata_normalized_text = unicodedata.normalize("NFKC", text)
        return lemmatize(unicodedata_normalized_text)

    def vectorize(self, texts):
        self.raw_vectorizer = CountVectorizer(tokenizer=self._unicodedata_token)
        self.raw_vectorizer.fit(texts)
        return self.raw_vectorizer

    def lemmatize(self, texts):
        self.lemmatized_vectorizer = CountVectorizer(tokenizer=self._unicodedata_normalized_token)
        self.lemmatized_vectorizer.fit(texts)
        return self.lemmatized_vectorizer

    def raw_tokenize(self, text):
        return tokenize(text)

    def raw_lemmatize(self, text):
        return lemmatize(text)

    def neologdn_token(self, text):
        neologdn_normalized_text = neologdn.normalize(text)
        return tokenize(neologdn_normalized_text)

    def neologdn_normalized_token(self, text):
        neologdn_normalized_text = neologdn.normalize(text)
        return lemmatize(neologdn_normalized_text)

    def raw_bow(self, texts):
        return self.raw_vectorizer.transform(texts).toarray()

    def raw_vocabulary(self, texts):
        return self.raw_vectorizer.vocabulary_

    def lemmatized_bow(self, texts):
        return self.lemmatized_vectorizer.transform(texts).toarray()

    def lemmatized_vocabulary(self, texts):
        return self.lemmatized_vectorizer.vocabulary_
