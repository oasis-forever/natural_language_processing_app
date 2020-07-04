import unittest
import sys
sys.path.append("../lib")
from mecab_parse import Mecab
import MeCab
import contextlib

class TestVector(unittest.TestCase):
    def setUp(self):
        self.mecab = Mecab()

    def test_parse(self):
        self.assertEqual("私\tワタクシ\tワタクシ\t私-代名詞\t代名詞\t\t\t0\nは\tワ\tハ\tは\t助詞-係助詞\t\t\t\nサーバー\tサーバー\tサーバー\tサーバー-server\t名詞-普通名詞-一般\t\t\t0,1\nサイド\tサイド\tサイド\tサイド-side\t名詞-普通名詞-一般\t\t\t1\nエンジニア\tエンジニア\tエンジニア\tエンジニア-engineer\t名詞-普通名詞-一般\t\t\t3\nです\tデス\tデス\tです\t助動詞\t助動詞-デス\t終止形-一般\t\nEOS\n", self.mecab.parse("私はサーバーサイドエンジニアです"))

    def _calFUT1(self):
        return self.mecab.parse_to_node_surface("私はサーバーサイドエンジニアです")

    def test_parse_to_node_surface(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT1()

        actual = buf.getvalue()
        self.assertEqual("\n私\nは\nサーバー\nサイド\nエンジニア\nです\n\n", actual)

    def _calFUT2(self):
        return self.mecab.parse_to_node_feature("私はサーバーサイドエンジニアです")

    def test_parse_to_node_feature(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT2()

        actual = buf.getvalue()
        self.assertEqual("BOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*\n代名詞,*,*,*,*,*,ワタクシ,私-代名詞,私,ワタクシ,私,ワタクシ,和,*,*,*,*,ワタクシ,ワタクシ,ワタクシ,ワタクシ,*,*,0,*,*\n助詞,係助詞,*,*,*,*,ハ,は,は,ワ,は,ワ,和,*,*,*,*,ハ,ハ,ハ,ハ,*,*,*,\"動詞%F2@0,名詞%F1,形容詞%F2@-1\",*\n名詞,普通名詞,一般,*,*,*,サーバー,サーバー-server,サーバー,サーバー,サーバー,サーバー,外,*,*,*,*,サーバー,サーバー,サーバー,サーバー,*,*,\"0,1\",C2,*\n名詞,普通名詞,一般,*,*,*,サイド,サイド-side,サイド,サイド,サイド,サイド,外,*,*,*,*,サイド,サイド,サイド,サイド,*,*,1,C1,*\n名詞,普通名詞,一般,*,*,*,エンジニア,エンジニア-engineer,エンジニア,エンジニア,エンジニア,エンジニア,外,*,*,*,*,エンジニア,エンジニア,エンジニア,エンジニア,*,*,3,C1,*\n助動詞,*,*,*,助動詞-デス,終止形-一般,デス,です,です,デス,です,デス,和,*,*,*,*,デス,デス,デス,デス,*,*,*,\"形容詞%F2@-1,動詞%F2@0,名詞%F2@1\",*\nBOS/EOS,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*\n", actual)

if __name__ == "__main__":
    unittest.main()
