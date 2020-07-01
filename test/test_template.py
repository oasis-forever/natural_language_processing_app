import unittest
import sys
sys.path.append("../lib")
from module import Module

class TestModule(unittest.TestCase):
    # Return pi
    def test_method(self):
        self.assertEqual(expected_val, module.method())

if __name__ == "__main__":
    unittest.main()
