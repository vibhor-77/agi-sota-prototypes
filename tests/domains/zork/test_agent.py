import unittest
import sys
import os
from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.agent import ZorkDeepAgent

class TestZorkAgent(unittest.TestCase):
    def test_bfs_graph_builder(self):
        rom_path = os.path.join('data', 'zork', 'zork1.z5')
        if not os.path.exists(rom_path):
            self.skipTest("Zork ROM not downloaded.")
            
        agent = ZorkDeepAgent()
        
        # Mute prints for testing
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            agent.explore_world(lambda: ZorkSOTAEnvironment(level=1), max_depth=1)
            sys.stdout = sys.__stdout__
            
        # Should map the root node and its immediate branches
        self.assertTrue(len(agent.known_states) >= 4)
        
        env = ZorkSOTAEnvironment(level=1)
        plan = agent.search(env, target_keyword="leaflet")
        self.assertIsNotNone(plan)
        self.assertEqual(plan, ["open mailbox"])

if __name__ == '__main__':
    unittest.main()
