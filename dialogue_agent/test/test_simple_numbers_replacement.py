import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
import simple_numbers_replacement as s

class TestSimpleNumberReplacement(unittest.TestCase):
    def setUp(self):
        self.text1 = "本を1冊読んだ"
        self.text2 = "本を10冊読んだ"
        self.text3 = "本を100冊読んだ"

    def test_tokenize_numbers1(self):
        self.assertEqual("本を SOMEBUNBER 冊読んだ", s.tokenize_numbers(self.text1))

    def test_tokenize_numbers2(self):
        self.assertEqual("本を SOMEBUNBER 冊読んだ", s.tokenize_numbers(self.text2))

    def test_tokenize_numbers3(self):
        self.assertEqual("本を SOMEBUNBER 冊読んだ", s.tokenize_numbers(self.text3))

if __name__ == "__main__":
    unittest.main()
