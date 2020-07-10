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
        self.input_text = "名前を教えて下さい"
        self.replies = "../csv/replies.csv"

    def _calFUT(self):
        return self.dialogue_agent.reply(self.input_text, self.replies)

    def test_reply(self):
        from io import StringIO
        buf = StringIO()

        with contextlib.redirect_stdout(buf):
            self._calFUT()

        actual = buf.getvalue()
        self.assertEqual("私はOasistといいます\n", actual)

if __name__ == "__main__":
    unittest.main()
