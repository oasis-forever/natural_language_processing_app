import unittest
import sys
sys.path.append("../lib")
from mecab_parse import Mecab
import MeCab
import contextlib

class TestVector(unittest.TestCase):
    def setUp(self):
        self.mecab = Mecab()
        self.text = "私はサーバーサイドエンジニアです"

    def test_parse(self):
        self.assertEqual("私\t名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ\nは\t助詞,係助詞,*,*,*,*,は,ハ,ワ\nサーバーサイドエンジニア\t名詞,一般,*,*,*,*,*\nです\t助動詞,*,*,*,特殊・デス,基本形,です,デス,デス\nEOS\n", self.mecab.parse(self.text))

    def _calFUT1(self):
        return self.mecab.parse_to_node_surface(self.text)

    def test_parse_to_node_surface(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT1()

        actual = buf.getvalue()
        self.assertEqual("\n私\nは\nサーバーサイドエンジニア\nです\n\n", actual)

    def _calFUT2(self):
        return self.mecab.parse_to_node_feature(self.text)

    def test_parse_to_node_feature(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT2()

        actual = buf.getvalue()
        self.assertEqual("BOS/EOS,*,*,*,*,*,*,*,*\n名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ\n助詞,係助詞,*,*,*,*,は,ハ,ワ\n名詞,一般,*,*,*,*,*\n助動詞,*,*,*,特殊・デス,基本形,です,デス,デス\nBOS/EOS,*,*,*,*,*,*,*,*\n", actual)

    def test_tokenize(self):
        self.assertEqual(["私", "は", "サーバーサイドエンジニア", "です"], self.mecab.tokenize(self.text))

if __name__ == "__main__":
    unittest.main()
