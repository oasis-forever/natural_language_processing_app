import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from dialogue_agent import DialogueAgent

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        self.dialogue_agent = DialogueAgent()
        self.dialogue_agent.train()
        input_text = "名前を教えて下さい"
        self.dialogue_agent.predict([input_text])

    def test_reply(self):
        self.assertEqual("私はOasistといいます", self.dialogue_agent.reply())

if __name__ == "__main__":
    unittest.main()
