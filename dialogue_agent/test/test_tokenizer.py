import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
import tokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.text1 = "本を読んだ"
        self.text2 = "本を読みました"

    def test_tokenize1(self):
        self.assertEqual(["本", "を", "読ん", "だ"], tokenizer.tokenize(self.text1))

    def test_tokenize2(self):
        self.assertEqual(["本", "を", "読み", "まし", "た"], tokenizer.tokenize(self.text2))

    def test_lemmatize1(self):
        self.assertEqual(["本", "を", "読む", "だ"], tokenizer.lemmatize(self.text1))

    def test_lemmatize2(self):
        self.assertEqual(["本", "を", "読む", "ます", "た"], tokenizer.lemmatize(self.text2))

    def test_remove_stop_words1(self):
        self.assertEqual(["本", "を", "読む", "だ"], tokenizer.remove_stop_words(self.text1))

    def test_remove_stop_words2(self):
        self.assertEqual(["本", "を", "読む", "ます", "た"], tokenizer.remove_stop_words(self.text2))

    def test_remove_auxiliary_verbs_and_particles1(self):
        self.assertEqual(["本", "読む"], tokenizer.remove_auxiliary_verbs_and_particles(self.text1))

    def test_remove_auxiliary_verbs_and_particles2(self):
        self.assertEqual(["本", "読む"], tokenizer.remove_auxiliary_verbs_and_particles(self.text2))

if __name__ == "__main__":
    unittest.main()
