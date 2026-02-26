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

    def explore_world(self, env_class, max_depth=12):
        print("[AGENT] Exploring phase: Deduce rules, items, and dependencies (Systematic BFS)...")
        base_actions = [
            "go north", "go south", "go east", "go west",
            "take sword", "take lamp", "take key", "take gold",
            "open door", "attack troll with sword"
        ]
        
        queue = deque([([], [])])
        visited_paths = set()
        
        while queue:
            path, current_inv = queue.popleft()
            if len(path) > max_depth:
                continue
                
            for action in base_actions:
                # 1. Replay path to reach state
                env = env_class()
                env_inv = []
                obs = env.get_observation()
                for past_act in path:
                    sem = self.parser.parse(past_act)
                    action_result = env.execute_action(sem)
                    obs = f"{action_result} | {env.get_observation()}"
                    if "picked up" in action_result: env_inv.append(sem["target"])
                
                s_t = self._hash_state(obs, env_inv)
                self.known_states.add(s_t)
                
                # 2. Try the new action
                sem = self.parser.parse(action)
                action_result = env.execute_action(sem)
                obs_next = f"{action_result} | {env.get_observation()}"
                
                next_inv = list(env_inv)
                if "picked up" in action_result:
                    item_taken = sem.get("target")
                    if item_taken and item_taken not in next_inv:
                        next_inv.append(item_taken)
                        
                s_t_next = self._hash_state(obs_next, next_inv)
                self.known_states.add(s_t_next)
                
                if s_t not in self.world_graph:
                    self.world_graph[s_t] = {}
                    
                # 3. Only keep actions that actually do something
                if "cannot" not in obs_next and "don't understand" not in obs_next:
                    if obs_next != obs:
                        if action not in self.world_graph[s_t]:
                            self.world_graph[s_t][action] = s_t_next
                        if s_t_next not in visited_paths:
                            visited_paths.add(s_t_next)
                            queue.append((path + [action], next_inv))
                            
        print(f"[AGENT] Exploration Complete.")
        
    def search(self, start_obs, target_keyword, **kwargs):
        print(f"\n[AGENT] BFS Graph Search over composed hypotheses for: '{target_keyword}'")
        start_state = self._hash_state(start_obs, [])
        queue = deque([(start_state, [])])
        visited = {start_state}
        
        while queue:
            curr_state, path = queue.popleft()
            
            if target_keyword.lower() in curr_state.lower():
                return path
                
            if curr_state not in self.world_graph:
                continue
                
            for action_text, next_state in self.world_graph[curr_state].items():
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [action_text]))
                    
        return None
