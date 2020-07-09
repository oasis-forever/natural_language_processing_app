import unittest
import sys
sys.path.append("../lib")
from text import Text
import contextlib

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        self.text1 = Text("[初めてのTensorFlow]は定価2200円+税です")
        self.text2 = Text("[初めての　TensorFlow]は定価２２００円+税です")
        self.text3 = Text("[初めての TensorFlow]は定価2200円+税です")

    def test_raw_tokenize_text1(self):
        # FIXME: mecab-python3 does not provide a propper word devider, so even single-byte digits are counted as an element of list.
        # This probles seems not to be fixed yet: https://github.com/SamuraiT/mecab-python3/issues/19
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2", "2", "0", "0", "円", "+", "税", "です"], self.text1.raw_tokenize())

    def test_raw_tokenize_text2(self):
        self.assertEqual(["[", "初めて", "の", "\u3000", "TensorFlow", "]", "は", "定価", "２", "２", "０", "０", "円", "+", "税", "です"], self.text2.raw_tokenize())

    def test_raw_tokenize_text3(self):
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2", "2", "0", "0", "円", "+", "税", "です"], self.text3.raw_tokenize())

    def test_neologdn_normalized_token_text2(self):
        self.assertEqual(self.text1.raw_tokenize(), self.text2.neologdn_normalized_token())

    def test_neologdn_normalized_token_text3(self):
        self.assertEqual(self.text1.raw_tokenize(), self.text3.neologdn_normalized_token())

if __name__ == "__main__":
    unittest.main()
