import json
import os
from core.feedback import Environment

class ZorkSOTAEnvironment(Environment):
    """
    Pillar 1: Feedback Loop.
    A wrapper around Jericho official Zork I ROM.
    """
    def __init__(self, level=3):
        self.level = level
        rom_path = os.path.join('data', 'zork', 'zork1.z5')
        
        try:
            import jericho
        except ImportError:
            raise ImportError("Please install jericho: pip install jericho")
            
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"Zork ROM not found at {rom_path}")
            
        # Initialize Jericho with the official Zork 1 binary
        self.env = jericho.FrotzEnv(rom_path)
        self.last_obs, _ = self.env.reset()
        
    def get_observation(self):
        return self.last_obs

    def get_score(self):
        return self.env.get_score()

    def get_state(self):
        return self.env.get_state()
        
    def load_state(self, state):
        self.env.set_state(state)
        
    def get_valid_actions(self):
        # Disable parallel validation to prevent leaking multiprocessing pools (improves stability on Mac/Python 3.9)
        return self.env.get_valid_actions(use_parallel=False)
        
    def step_raw(self, action_string):
        """Immediate execution for BFS loops without semantic parsing overhead"""
        obs, reward, done, info = self.env.step(action_string)
        self.last_obs = obs.strip()
        return self.last_obs

    def execute_action(self, action_dict):
        verb = action_dict.get("action", "").lower()
        target = action_dict.get("target", "").lower()
        tool = action_dict.get("tool", "")
        
        # Reconstruct natural language for the Frotz engine
        command = verb
        if target: command += f" {target}"
        if tool: command += f" with {tool}"
        
        return self.step_raw(command)
