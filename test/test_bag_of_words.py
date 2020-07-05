import unittest
import sys
sys.path.append("../lib")
from bag_of_words import BagOfWords
from mecab_parse import Mecab
import MeCab
import contextlib

class TestVector(unittest.TestCase):
    def setUp(self):
        self.mecab = Mecab()
        self.bag_of_words = BagOfWords()

    def test_calc_bow(self):
        texts = [
            "私は私のことが好きなあなたが好きです",
            "私はラーメンが好きです",
            "富士山は日本一高い山です",
        ]
        tokenized_texts = [self.mecab.tokenize(text) for text in texts]
        vocabulary, bow = self.bag_of_words.calc_bow(tokenized_texts)
        self.assertEqual(
            {
                "私": 0,
                "は": 1,
                "の": 2,
                "こと": 3,
                "が": 4,
                "好き": 5,
                "な": 6,
                "あなた": 7,
                "です": 8,
                "ラーメン": 9,
                "富士": 10,
                "山": 11,
                "日本": 12,
                "一": 13,
                "高い": 14
            },
            vocabulary
        )
        self.assertEqual(
            [
                [2, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1]
            ],
            bow
        )

if __name__ == "__main__":
    unittest.main()
