import MeCab
import sys
sys.path.append("./concern")
import neologdn
import unicodedata

tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

def _preprocess(text):
    text = unicodedata.normalize("NFKC", text)
    text = neologdn.normalize(text)
    text = text.lower()
    return text

def tokenize(text):
    node = tagger.parseToNode(_preprocess(text))
    result = []
    while node:
        features = node.feature.split(",")
        if features[0] != "BOS/EOS":
            if features[0] not in ["助詞", "助動詞"]:
                token = features[6] if features[6] != "*" else node.surface
                result.append(token)
        node = node.next
    return result