import unittest
import os
from domains.zork.env import ZorkSOTAEnvironment

class TestZorkEnv(unittest.TestCase):
    def test_environment_initialization(self):
        # Only run if we actually downloaded the ROM (handled in our setup)
        rom_path = os.path.join('data', 'zork', 'zork1.z5')
        if not os.path.exists(rom_path):
            self.skipTest("Zork ROM not downloaded yet, skipping.")
            
        env = ZorkSOTAEnvironment(level=1)
        obs = env.get_observation()
        self.assertTrue("West of House" in obs)
        
    def test_environment_step(self):
        env = ZorkSOTAEnvironment(level=1)
        obs = env.execute_action({"action": "open", "target": "mailbox"})
        self.assertTrue("Opening the small mailbox reveals a leaflet." in obs)

    def test_raw_valid_actions(self):
        env = ZorkSOTAEnvironment(level=1)
        actions = env.get_valid_actions()
        self.assertIn("open mailbox", actions)
        
    def test_state_saving_loading(self):
        env = ZorkSOTAEnvironment(level=1)
        start_state = env.get_state()
        
        env.step_raw("open mailbox")
        self.assertNotIn("open mailbox", env.get_valid_actions()) # already open
        
        env.load_state(start_state)
        self.assertIn("open mailbox", env.get_valid_actions()) # back to start

if __name__ == '__main__':
    unittest.main()
