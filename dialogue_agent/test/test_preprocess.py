import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from preprocess import PreProcess
import numpy as np
from numpy.testing import assert_array_equal

class TestPreProcess(unittest.TestCase):
    def setUp(self):
        self.preprocess = PreProcess()
        self.text1 = "[初めてのTensorFlow]は定価2200円+税です"
        self.text2 = "[初めての　TensorFlow]は定価２２００円+税です"
        self.text3 = "[初めての TensorFlow]は定価2200円+税です"
        self.texts = ["㈱自然言語処理研究に入社した", "(株)自然言語処理研究に入社した"]

    def test_raw_tokenize_text1(self):
        # FIXME: mecab-python3 does not provide a propper word devider, so even single-byte digits are counted as an element of list.
        # This probles seems not to be fixed yet: https://github.com/SamuraiT/mecab-python3/issues/19
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2200円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text1))

    def test_raw_tokenize_text2(self):
        self.assertEqual(["[", "初めて", "の", "\u3000", "TensorFlow", "]", "は", "定価", "２２００円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text2))

    def test_raw_tokenize_text3(self):
        self.assertEqual(["[", "初めて", "の", "TensorFlow", "]", "は", "定価", "2200円", "+", "税", "です"], self.preprocess.raw_tokenize(self.text3))

    def test_raw_lemmatize_text1(self):
        # FIXME: mecab-python3 does not provide a propper word devider, so even single-byte digits are counted as an element of list.
        # This probles seems not to be fixed yet: https://github.com/SamuraiT/mecab-python3/issues/19
        self.assertEqual(["[", "初めて", "TensorFlow", "]", "定価", "2200円", "+", "税"], self.preprocess.raw_lemmatize(self.text1))

    def test_raw_lemmatize_text2(self):
        self.assertEqual(["[", "初めて", "TensorFlow", "]", "定価", "2200円", "+", "税"], self.preprocess.raw_lemmatize(self.text2))

    def test_raw_lemmatize_text3(self):
        self.assertEqual(["[", "初めて", "TensorFlow", "]", "定価", "2200円", "+", "税"], self.preprocess.raw_lemmatize(self.text3))

    def test_neologdn_token_text1(self):
        self.assertEqual(self.preprocess.raw_tokenize(self.text1), self.preprocess.neologdn_token(self.text2))

    def test_neologdn_token_text2(self):
        self.assertEqual(self.preprocess.raw_tokenize(self.text1), self.preprocess.neologdn_token(self.text3))

    def test_neologdn_normalized_token_text1(self):
        self.assertEqual(self.preprocess.raw_lemmatize(self.text1), self.preprocess.neologdn_normalized_token(self.text2))

    def test_neologdn_normalized_token_text2(self):
        self.assertEqual(self.preprocess.raw_lemmatize(self.text1), self.preprocess.neologdn_normalized_token(self.text3))

    def test_raw_bow(self):
        assert_array_equal(
            np.array([
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]
            ]), self.preprocess.raw_bow(self.texts)
        )

    def test_lemmatized_vocaburaly(self):
        self.assertEqual({"(株)": 0, "自然言語処理": 6, "研究": 5, "に": 3, "入社": 4, "し": 1, "た": 2}, self.preprocess.raw_vocabulary(self.texts))

    def test_lemmatized_bow(self):
        assert_array_equal(
            np.array([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]), self.preprocess.lemmatized_bow(self.texts)
        )

    def test_lemmatized_vocaburaly(self):
        self.assertEqual({"(株)": 0, "自然言語処理": 4, "研究": 3, "入社": 2, "する": 1}, self.preprocess.lemmatized_vocabulary(self.texts))

if __name__ == "__main__":
    unittest.main()
