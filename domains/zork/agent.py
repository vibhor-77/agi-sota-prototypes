from collections import deque
import heapq
import pickle
from core.exploration import SearchAlgorithm

class ZorkDeepAgent(SearchAlgorithm):
    """
    Pillar 4: Exploration.
    Systematic pathfinding using Breadth First Search to resolve deep state dependencies.
    """
    def __init__(self):
        self.world_graph = {}
        self.known_states = set()
        
    def _hash_state(self, obs, inventory):
        inv_str = ",".join(sorted(inventory))
        return f"OBS:[{obs}]|INV:[{inv_str}]"
        
    def _state_to_hash(self, state):
        return pickle.dumps(state)

    def explore_world(self, env_class, max_depth=12, max_states=5000):
        print(f"[AGENT] A* Best-First Search (Depth: {max_depth}, Budget: {max_states} states)...")
        
        env = env_class()
        self.state_texts = {}
        
        start_state = env.get_state()
        start_hash = self._state_to_hash(start_state)
        self.known_states.add(start_hash)
        self.state_texts[start_hash] = env.get_observation()
        
        # Priority Queue: (-score, depth, path, state_hash, state_obj)
        # Negative score because heapq is a min-heap — higher scores pop first.
        pq = []
        heapq.heappush(pq, (0, 0, [], start_hash, start_state))
        
        best_score_found = 0
        states_expanded = 0
        
        while pq and states_expanded < max_states:
            neg_score, depth, path, curr_hash, curr_state = heapq.heappop(pq)
            current_score = -neg_score
            states_expanded += 1
            
            if current_score > best_score_found:
                best_score_found = current_score
                print(f"[AGENT] Score ↑ {best_score_found}/350 at depth {depth} ({states_expanded} states)")
            
            # Progress logging every 500 states
            if states_expanded % 500 == 0:
                print(f"[AGENT] ... {states_expanded}/{max_states} states | Best: {best_score_found}/350 | Queue: {len(pq)}")
                
            if depth >= max_depth:
                continue
                
            env.load_state(curr_state)
            valid_actions = env.get_valid_actions()
            
            for action in valid_actions:
                env.load_state(curr_state)
                obs = env.step_raw(action)
                next_score = env.get_score()
                
                next_state = env.get_state()
                next_hash = self._state_to_hash(next_state)
                
                if curr_hash not in self.world_graph:
                    self.world_graph[curr_hash] = {}
                    
                self.world_graph[curr_hash][action] = next_hash
                
                if next_hash not in self.known_states:
                    self.known_states.add(next_hash)
                    self.state_texts[next_hash] = obs
                    
                    heapq.heappush(pq, (-next_score, depth + 1, path + [action], next_hash, next_state))
                    
        env.load_state(start_state)
        print(f"[AGENT] Done. {states_expanded} expanded, {len(self.known_states)} unique states. Best: {best_score_found}/350")
        
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
