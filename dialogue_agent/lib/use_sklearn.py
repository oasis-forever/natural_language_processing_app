from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenize

class UseSkLearn():
    def __init__(self):
        pass

    def calc_bow(self, texts):
        vectorizer = CountVectorizer(tokenizer=tokenize)
        vectorizer.fit(texts)
        bow = vectorizer.transform(texts)
        return bow.toarray()
