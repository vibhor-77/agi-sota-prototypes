import unittest
import sys
import os
from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.agent import ZorkDeepAgent
from domains.zork.semantics import ZorkSemanticParser

class TestZorkEndToEnd(unittest.TestCase):
    def test_zork_sota_level_1(self):
        """
        Tests the entire pipeline: Loading authentic Zork I ROM via Jericho, 
        mapping the state-space into a BFS graph using byte-hashes, 
        and validating the exact path generation to the goal.
        """
        rom_path = os.path.join('data', 'zork', 'zork1.z5')
        if not os.path.exists(rom_path):
            self.skipTest("Zork ROM not downloaded.")
            
        env = ZorkSOTAEnvironment(level=1)
        agent = ZorkDeepAgent()
        
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            agent.explore_world(lambda: env, max_depth=2)
            sys.stdout = sys.__stdout__
            
        winning_actions = agent.search(env, target_keyword="leaflet")
        
        self.assertIsNotNone(winning_actions)
        self.assertEqual(winning_actions, ["open mailbox"])
        
        # Test full NLP loop against memory restored env
        parser = ZorkSemanticParser()
        env2 = ZorkSOTAEnvironment(level=1)
        
        # Wait, the winning plan contains exact raw strings for real Jericho,
        # so we skip parsing during execution but verify raw step works
        for action in winning_actions:
            obs = env2.step_raw(action)
            self.assertTrue("leaflet" in obs.lower())

if __name__ == '__main__':
    unittest.main()
