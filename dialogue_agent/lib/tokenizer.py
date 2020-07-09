import MeCab

tagger = MeCab.Tagger()

def tokenize(text):
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens

def lemmatize(text):
    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(",")
        if features[0] != "BOS/EOS":
            # assign index word or non-lemmatize word
            token = features[7] if features[7] != "*" else node.surface
            result.append(token)
        node = node.next
    return result
