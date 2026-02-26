import unittest
from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.semantics import ZorkSemanticParser
from domains.zork.agent import ZorkDeepAgent

class TestZorkDomain(unittest.TestCase):
    def setUp(self):
        self.parser = ZorkSemanticParser()

    def test_semantic_parser(self):
        sem1 = self.parser.parse("go north")
        self.assertEqual(sem1, {"action": "MOVE", "target": "NORTH", "tool": None})
        sem2 = self.parser.parse("take the sword please")
        self.assertEqual(sem2, {"action": "TAKE", "target": "sword", "tool": None})
        sem3 = self.parser.parse("attack the troll with my sword")
        self.assertEqual(sem3, {"action": "ATTACK", "target": "troll", "tool": "sword"})
        
        # Edge cases/Adversarial
        sem4 = self.parser.parse("I don't want to go down, let's head east towards the door")
        self.assertEqual(sem4["action"], "MOVE")
        self.assertEqual(sem4["target"], "EAST")
        
        sem5 = self.parser.parse("grab the glowing lamp quickly")
        self.assertEqual(sem5["action"], "TAKE")
        self.assertEqual(sem5["target"], "lamp")

    def test_env_level_1_constraints(self):
        env = ZorkSOTAEnvironment(level=1)
        # Can't enter without key
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("locked" in obs.lower())
        # Get key and open
        env.execute_action({"action": "TAKE", "target": "key"})
        env.execute_action({"action": "OPEN", "target": "door", "tool": "key"})
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("treasure room" in obs.lower())
        
    def test_env_level_2_constraints(self):
        env = ZorkSOTAEnvironment(level=2)
        # Try to walk past troll
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("block" in obs.lower() or "cannot" in obs.lower())
        # Grab sword and kill
        env.execute_action({"action": "TAKE", "target": "sword"})
        obs = env.execute_action({"action": "ATTACK", "target": "troll", "tool": "sword"})
        self.assertTrue("slay" in obs.lower())
        # Enter
        obs = env.execute_action({"action": "MOVE", "target": "EAST"})
        self.assertTrue("treasure" in obs.lower())

    def test_env_level_3_constraints(self):
        env = ZorkSOTAEnvironment(level=3)
        env.execute_action({"action": "MOVE", "target": "NORTH"})
        # Should be too dark to see the key
        obs = env.execute_action({"action": "TAKE", "target": "key"})
        self.assertTrue("too dark" in obs.lower())
        
    def test_bfs_graph_builder(self):
        agent = ZorkDeepAgent()
        
        # Mute prints for testing
        import sys, os
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            agent.explore_world(lambda: ZorkSOTAEnvironment(level=1), max_depth=3)
            sys.stdout = sys.__stdout__
        
        # Level 1 is simple, should map completely inside 3 steps.
        # Ensure it found multiple states
        self.assertTrue(len(agent.known_states) > 5)
        
        # Ensure it can plan the path
        env = ZorkSOTAEnvironment(level=1)
        plan = agent.search(env.get_observation(), target_keyword="picked up the gold")
        self.assertIsNotNone(plan)
        self.assertTrue("take key" in plan)

if __name__ == '__main__':
    unittest.main()
