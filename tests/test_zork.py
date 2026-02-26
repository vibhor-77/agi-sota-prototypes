import unittest
from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.semantics import ZorkSemanticParser

class TestZorkDomain(unittest.TestCase):
    def setUp(self):
        self.parser = ZorkSemanticParser()

    def test_semantic_parser(self):
        # Test Entity extraction and action mapping
        sem1 = self.parser.parse("go north")
        self.assertEqual(sem1, {"action": "MOVE", "target": "NORTH", "tool": None})
        
        sem2 = self.parser.parse("take the sword please")
        self.assertEqual(sem2, {"action": "TAKE", "target": "sword", "tool": None})
        
        sem3 = self.parser.parse("attack the troll with my sword")
        self.assertEqual(sem3, {"action": "ATTACK", "target": "troll", "tool": "sword"})
        
        sem4 = self.parser.parse("unlock the heavy oak door using the key")
        self.assertEqual(sem4, {"action": "OPEN", "target": "door", "tool": "key"})

    def test_env_level_1_constraints(self):
        # Level 1: Just a locked door. Need key.
        env = ZorkSOTAEnvironment(level=1)
        
        # Can't enter without key
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("locked" in obs.lower())
        
        # Get key and open
        env.execute_action({"action": "TAKE", "target": "key"})
        self.assertTrue("key" in env.state["inventory"])
        
        obs = env.execute_action({"action": "OPEN", "target": "door", "tool": "key"})
        self.assertTrue("unlock" in obs.lower())
        
        # Now can enter
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("treasure room" in obs.lower())

    def test_env_level_3_constraints(self):
        # Level 3: Dark, troll, locked.
        env = ZorkSOTAEnvironment(level=3)
        
        env.execute_action({"action": "MOVE", "target": "NORTH"})
        # Should be too dark to see the lamp
        obs = env.execute_action({"action": "TAKE", "target": "key"})
        self.assertTrue("too dark" in obs.lower())
        
        # But can grab lamp
        env.execute_action({"action": "TAKE", "target": "lamp"})
        self.assertTrue("lamp" in env.state["inventory"])
        
        # Now can see the troll. Try fighting without sword
        obs = env.execute_action({"action": "ATTACK", "target": "troll"})
        self.assertTrue("bare-handed and die" in obs.lower() or "cannot" in obs.lower())

if __name__ == '__main__':
    unittest.main()
