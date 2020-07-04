import unittest
import sys
sys.path.append("../lib")
from mecab_parse import Mecab
import MeCab

class TestVector(unittest.TestCase):
    def setUp(self):
        self.mecab = Mecab()

    def test_parse(self):
        self.assertEqual("私\tワタクシ\tワタクシ\t私-代名詞\t代名詞\t\t\t0\nは\tワ\tハ\tは\t助詞-係助詞\t\t\t\nサーバー\tサーバー\tサーバー\tサーバー-server\t名詞-普通名詞-一般\t\t\t0,1\nサイド\tサイド\tサイド\tサイド-side\t名詞-普通名詞-一般\t\t\t1\nエンジニア\tエンジニア\tエンジニア\tエンジニア-engineer\t名詞-普通名詞-一般\t\t\t3\nです\tデス\tデス\tです\t助動詞\t助動詞-デス\t終止形-一般\t\nEOS\n", self.mecab.parse("私はサーバーサイドエンジニアです"))

if __name__ == "__main__":
    unittest.main()
