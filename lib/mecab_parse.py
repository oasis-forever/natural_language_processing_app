import MeCab

class Mecab:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def parse(self, text):
        return self.tagger.parse(text)

    def parse_to_node_surface(self, text):
        node = self.tagger.parseToNode(text)
        while node:
            print(node.surface)
            node = node.next

    def parse_to_node_feature(self, text):
        node = self.tagger.parseToNode(text)
        while node:
            print(node.feature)
            node = node.next

    def tokenize(self, text):
        tokens = []
        node = self.tagger.parseToNode(text)
        while node:
            if node.surface != "":
                tokens.append(node.surface)
            node = node.next
        return tokens