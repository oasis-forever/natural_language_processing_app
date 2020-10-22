from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("./concerns")
from lemmatizer import lemmatize

class SkLearnTfIdf():
    def __init__(self):
        pass

    def calc_tfidf(self, texts):
        vectorizer = TfidfVectorizer(tokenizer=lemmatize)
        vectorizer.fit(texts)
        tfidf = vectorizer.transform(texts)
        return tfidf.toarray()
