import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from tokenizer import tokenize

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.text1 = "本を読んだ"
        self.text2 = "本を読みました"

    def test_tokenize1(self):
        self.assertEqual(["本", "を", "読ん", "だ"], tokenize(self.text1))

    def test_tokenize2(self):
        self.assertEqual(["本", "を", "読み", "まし", "た"], tokenize(self.text2))

if __name__ == "__main__":
    unittest.main()
