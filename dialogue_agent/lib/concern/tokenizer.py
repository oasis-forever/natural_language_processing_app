import MeCab
import sys
sys.path.append("./concern")
from stop_words import stop_words

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

def remove_stop_words(text):
    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(",")
        if features[0] != "BOS/EOS":
            token = features[7] if features[7] != "*" else node.surface
            if token not in stop_words():
                result.append(token)
        node = node.next
    return result

def remove_auxiliary_verbs_and_particles(text):
    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(",")
        if features[0] != "BOS/EOS":
                if features[0] not in ["助詞", "助動詞"]:
                    token = features[7] if features[7] != "*" else node.surface
                    result.append(token)
        node = node.next
    return result
