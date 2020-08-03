import unittest
import sys
sys.path.append("../lib")
sys.path.append("../ja")
from analogy_task_ja import AnalogyTaskJa

class TestAnalogyTaskJa(unittest.TestCase):
    def setUp(self):
        self.analogy_task_ja = AnalogyTaskJa()
        self.analogy_task_ja.word_embeddings()
        self.analogy_task_ja.calc_vector()

    def test_closest_similar_words(self):
        self.assertEqual([
            ('パリ', 0.5090547800064087),
            ('東京', 0.4823923110961914),
            ('ルーアン', 0.46184849739074707),
            ('アムステルダム', 0.4602998197078705),
            ('ウィーン', 0.4559346139431),
            ('クラクフ', 0.4543963074684143),
            ('ブリュッセル', 0.4534413516521454),
            ('サンクトペテルブルク', 0.45139601826667786),
            ('ストラスブール', 0.4500630497932434),
            ('リヨン', 0.44288450479507446)], self.analogy_task_ja.similar_words(self.analogy_task_ja.vector)
        )

if __name__ == "__main__":
    unittest.main()
