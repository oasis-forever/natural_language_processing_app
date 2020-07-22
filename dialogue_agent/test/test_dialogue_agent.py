import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from dialogue_agent import DialogueAgent
import contextlib

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        training_data = "../csv/training_data.csv"
        self.dialogue_agent = DialogueAgent(training_data)
        self.dialogue_agent.extract_trainig_data(training_data)

    def _calFUT(self):
        input_text = "名前を教えて下さい"
        replies = "../csv/replies.csv"
        ngram_range = (1, 3)
        self.dialogue_agent.train(ngram_range)
        return self.dialogue_agent.reply(input_text, replies)

    def test_reply(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT()

        actual = buf.getvalue()
        self.assertEqual("私はOasistといいます\n", actual)

if __name__ == "__main__":
    unittest.main()
