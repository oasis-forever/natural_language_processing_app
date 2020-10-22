import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from keras_scikit_learn_api import KerasScikitLearnApi

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        self.ksla = KerasScikitLearnApi()
        self.ksla.fit_classifier()

    def test_reply(self):
        self.assertEqual([1], self.ksla.predict(some_feature))

if __name__ == "__main__":
    unittest.main()
