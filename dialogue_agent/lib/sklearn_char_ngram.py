from sklearn.feature_extraction.text import CountVectorizer

class SkLearnCharNgram():
    def __init__(self):
        pass

    def vectorize(self, texts, ngram_range):
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range)
        self.vectorizer.fit(texts)
        self.bow = self.vectorizer.transform(texts)

    def bow_array(self):
        return self.bow.toarray()

    def vocabulary(self):
        return self.vectorizer.vocabulary_
