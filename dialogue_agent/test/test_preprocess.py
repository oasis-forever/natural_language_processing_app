import unittest
import sys
sys.path.append("../lib")
from preprocess import PreProcess
import numpy as np

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        self.preprocess = PreProcess()
        self.text1 = "[初めてのTensorFlow]は定価2200円+税です"
        self.text2 = "[初めての　TensorFlow]は定価２２００円+税です"
        self.text3 = "[初めての TensorFlow]は定価2200円+税です"
        self.texts = ["㈱自然言語処理研究に入社した", "(株)自然言語処理研究に入社した"]

    def test_raw_tokenize_text1(self):
        # FIXME: mecab-python3 does not provide a propper word devider, so even single-byte digits are counted as an element of list.
        # This probles seems not to be fixed yet: https://github.com/SamuraiT/mecab-python3/issues/19
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2", "2", "0", "0", "円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text1))

    def test_raw_tokenize_text2(self):
        self.assertEqual(["[", "初めて", "の", "\u3000", "TensorFlow", "]", "は", "定価", "２", "２", "０", "０", "円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text2))

    def test_raw_tokenize_text3(self):
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2", "2", "0", "0", "円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text3))

    def test_neologdn_normalized_token_text2(self):
        self.assertEqual(self.preprocess.raw_tokenize(self.text1), self.preprocess.neologdn_normalized_token(self.text2))

    def test_neologdn_normalized_token_text3(self):
        self.assertEqual(self.preprocess.raw_tokenize(self.text1), self.preprocess.neologdn_normalized_token(self.text3))

    def test_bow(self):
        np.alltrue(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) == self.preprocess.bow(self.texts))

    def test_vocaburaly(self):
        self.assertEqual({'(': 0, '株': 7, ')': 1, '自然': 9, '言語': 10, '処理': 6, '研究': 8, 'に': 4, '入社': 5, 'し': 2, 'た': 3}, self.preprocess.vocabulary(self.texts))

if __name__ == "__main__":
    unittest.main()
