import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concern")
from dialogue_agent import DialogueAgent

class TestDialogueAgent(unittest.TestCase):
    def setUp(self):
        self.dialogue_agent = DialogueAgent()
        training_data = "../csv/training_data.csv"
        self.dialogue_agent.extract_trainig_data(training_data)
        self.dialogue_agent.train((1, 2))
        input_text = "名前を教えて下さい"
        self.dialogue_agent.predict([input_text])

    def test_reply(self):
        replies = "../csv/replies.csv"
        self.assertEqual("私はOasistといいます", self.dialogue_agent.reply(replies))

if __name__ == "__main__":
    unittest.main()
