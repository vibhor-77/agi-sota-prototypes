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

    def explore_world(self, env_class, max_depth=12, max_states=5000, verbose=False):
        print(f"[AGENT] A* Reward-Shaped Search (Depth: {max_depth}, Budget: {max_states} states)...")
        
        env = env_class()
        self.state_texts = {}
        self.best_path = []
        
        start_state = env.get_state()
        start_hash = self._state_to_hash(start_state)
        self.known_states.add(start_hash)
        self.state_texts[start_hash] = env.get_observation()
        
        unique_rooms = {env.get_observation()[:80]}
        
        start_score = env.get_score()
        try:
            start_inv = len(env.get_inventory())
            start_inv_items = env.get_inventory()
        except Exception:
            start_inv = 0
            start_inv_items = []
        start_h = start_score * 10 + start_inv * 2 + len(unique_rooms)
        
        if verbose:
            print(f"[VERBOSE] Start state: score={start_score}, inventory={start_inv_items}, rooms=1")
        
        pq = []
        heapq.heappush(pq, (-start_h, 0, [], start_hash, start_state))
        
        best_composite = start_h
        best_game_score = start_score
        states_expanded = 0
        prev_inv_count = start_inv
        
        while pq and states_expanded < max_states:
            neg_h, depth, path, curr_hash, curr_state = heapq.heappop(pq)
            current_h = -neg_h
            states_expanded += 1
            
            if current_h > best_composite:
                best_composite = current_h
                self.best_path = path
                
            if states_expanded % 500 == 0:
                print(f"[AGENT] ... {states_expanded}/{max_states} | h={best_composite} | Score: {best_game_score}/350 | Rooms: {len(unique_rooms)}")
                
            if depth >= max_depth:
                continue
                
            env.load_state(curr_state)
            valid_actions = env.get_valid_actions()
            
            if verbose and states_expanded <= 20:
                print(f"[VERBOSE] State #{states_expanded} depth={depth} h={current_h} | {len(valid_actions)} actions: {valid_actions[:5]}{'...' if len(valid_actions)>5 else ''}")
            
            for action in valid_actions:
                env.load_state(curr_state)
                obs = env.step_raw(action)
                next_game_score = env.get_score()
                
                try:
                    inv_items = env.get_inventory()
                    inv_count = len(inv_items)
                except Exception:
                    inv_items = []
                    inv_count = 0
                    
                room_sig = obs[:80]
                is_new_room = room_sig not in unique_rooms
                if is_new_room:
                    unique_rooms.add(room_sig)
                    if verbose:
                        print(f"[VERBOSE] 🗺️  NEW ROOM at depth {depth+1}: \"{room_sig}...\"")
                    
                composite_h = next_game_score * 10 + inv_count * 2 + len(unique_rooms)
                
                if next_game_score > best_game_score:
                    best_game_score = next_game_score
                    print(f"[AGENT] Score ↑ {best_game_score}/350 at depth {depth+1} via '{action}' ({states_expanded} states)")
                    if verbose:
                        print(f"[VERBOSE] 🎯 Path to score: {' → '.join(path + [action])}")
                
                if verbose and inv_count > prev_inv_count:
                    new_items = [i for i in inv_items if i not in start_inv_items]
                    print(f"[VERBOSE] 📦 INVENTORY CHANGE at depth {depth+1}: +{inv_count - prev_inv_count} items via '{action}' → {inv_items}")
                
                next_state = env.get_state()
                next_hash = self._state_to_hash(next_state)
                
                if curr_hash not in self.world_graph:
                    self.world_graph[curr_hash] = {}
                self.world_graph[curr_hash][action] = next_hash
                
                if next_hash not in self.known_states:
                    self.known_states.add(next_hash)
                    self.state_texts[next_hash] = obs
                    heapq.heappush(pq, (-composite_h, depth + 1, path + [action], next_hash, next_state))
                    
        env.load_state(start_state)
        print(f"[AGENT] Done. {states_expanded} expanded, {len(self.known_states)} states, {len(unique_rooms)} rooms. Best: {best_game_score}/350 (h={best_composite})")
        if verbose and self.best_path:
            print(f"[VERBOSE] Best path ({len(self.best_path)} steps): {' → '.join(self.best_path)}")
        
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
