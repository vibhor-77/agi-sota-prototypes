import unittest
from domains.zork.semantics import ZorkSemanticParser

class TestZorkSemantics(unittest.TestCase):
    def setUp(self):
        self.parser = ZorkSemanticParser()

    def test_semantic_parser_basic(self):
        sem1 = self.parser.parse("go north")
        self.assertEqual(sem1, {"action": "MOVE", "target": "NORTH", "tool": None})
        sem2 = self.parser.parse("take the sword please")
        self.assertEqual(sem2, {"action": "TAKE", "target": "sword", "tool": None})
        sem3 = self.parser.parse("attack the troll with my sword")
        self.assertEqual(sem3, {"action": "ATTACK", "target": "troll", "tool": "sword"})
        
    def test_semantic_parser_complex(self):
        # Edge cases/Adversarial
        sem4 = self.parser.parse("I don't want to go down, let's head east towards the door")
        self.assertEqual(sem4["action"], "MOVE")
        self.assertEqual(sem4["target"], "EAST")
        
        sem5 = self.parser.parse("grab the glowing lamp quickly")
        self.assertEqual(sem5["action"], "TAKE")
        self.assertEqual(sem5["target"], "lamp")

if __name__ == '__main__':
    unittest.main()
