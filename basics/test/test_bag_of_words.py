import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from bag_of_words import BagOfWords
from mecab_parse import Mecab
from tokenizer import tokenize

class TestBagOfWords(unittest.TestCase):
    def setUp(self):
        self.mecab = Mecab()
        self.bag_of_words = BagOfWords()
        self.texts = [
            "私は私のことが好きなあなたを愛しています",
            "私はラーメンが好きです",
            "富士山は日本一高い山です",
        ]

    def test_calc_bow(self):
        tokenized_texts = [tokenize(text) for text in self.texts]
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
                "を": 8,
                "愛し": 9,
                "て": 10,
                "い": 11,
                "ます": 12,
                "ラーメン": 13,
                "です": 14,
                "富士山": 15,
                "日本一": 16,
                "高い": 17,
                "山": 18
            },
            vocabulary
        )
        self.assertEqual(
            [
                [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ], bow
        )

    def test_calc_bow_counter_ver(self):
        tokenized_texts = [tokenize(text) for text in self.texts]
        bow = self.bag_of_words.calc_bow_counter_ver(tokenized_texts)
        self.assertEqual(
            [
                [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ], bow
        )

if __name__ == "__main__":
    unittest.main()
