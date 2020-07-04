import MeCab

class Mecab:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def parse(self, text):
        return self.tagger.parse(text)
