import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from lemmatizer import lemmatize

class TestLemmatizer(unittest.TestCase):
    def setUp(self):
        self.text1 = "本を読んだ"
        self.text2 = "本を読みました"

    def test_tokenize1(self):
        self.assertEqual(["本", "読む"], lemmatize(self.text1))

    def test_tokenize2(self):
        self.assertEqual(["本", "読む"], lemmatize(self.text2))

if __name__ == "__main__":
    unittest.main()
