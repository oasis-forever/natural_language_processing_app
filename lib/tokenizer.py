import MeCab

def tokenize(text):
    tokens = []
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens
