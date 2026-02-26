from collections import deque
from core.exploration import SearchAlgorithm
from domains.zork.semantics import ZorkSemanticParser

class ZorkDeepAgent(SearchAlgorithm):
    """
    Pillar 4: Exploration.
    Systematic pathfinding using Breadth First Search to resolve deep state dependencies.
    """
    def __init__(self):
        self.world_graph = {}
        self.known_states = set()
        self.parser = ZorkSemanticParser()
        
    def _hash_state(self, obs, inventory):
        inv_str = ",".join(sorted(inventory))
        return f"OBS:[{obs}]|INV:[{inv_str}]"
        
    def _state_to_hash(self, state):
        import pickle
        return hash(pickle.dumps(state))

    def explore_world(self, env_class, max_depth=12):
        print("[AGENT] Exploring phase: Memory-mapped BFS via Jericho...")
        
        env = env_class()
        self.state_texts = {}
        
        start_state = env.get_state()
        start_hash = self._state_to_hash(start_state)
        self.known_states.add(start_hash)
        self.state_texts[start_hash] = env.get_observation()
        
        queue = deque([(start_state, [])])
        
        while queue:
            curr_state, path = queue.popleft()
            if len(path) > max_depth:
                continue
                
            env.load_state(curr_state)
            valid_actions = env.get_valid_actions()
            
            for action in valid_actions:
                env.load_state(curr_state)
                obs = env.step_raw(action)
                
                next_state = env.get_state()
                next_hash = self._state_to_hash(next_state)
                
                curr_hash = self._state_to_hash(curr_state)
                if curr_hash not in self.world_graph:
                    self.world_graph[curr_hash] = {}
                    
                self.world_graph[curr_hash][action] = next_hash
                
                if next_hash not in self.known_states:
                    self.known_states.add(next_hash)
                    self.state_texts[next_hash] = obs
                    queue.append((next_state, path + [action]))
                    
        env.load_state(start_state)
        print(f"[AGENT] Exploration Complete.")
        
    def search(self, start_env, target_keyword, **kwargs):
        print(f"\n[AGENT] BFS Graph Search over composed hypotheses for: '{target_keyword}'")
        
        start_state = start_env.get_state()
        start_hash = self._state_to_hash(start_state)
        
        queue = deque([(start_hash, [])])
        visited = {start_hash}
        
        while queue:
            curr_hash, path = queue.popleft()
            
            obs_text = self.state_texts.get(curr_hash, "")
            if target_keyword.lower() in obs_text.lower():
                return path
                
            if curr_hash not in self.world_graph:
                continue
                
            for action_text, next_hash in self.world_graph[curr_hash].items():
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_hash, path + [action_text]))
                    
        return None
