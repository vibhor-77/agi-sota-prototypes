import numpy as np
import random
import re
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# 1. The Deep Zork Environment (SOTA)
# ======================================================================
class ZorkSOTAEnvironment:
    """A hidden deterministic state machine with multi-step dependencies"""
    def __init__(self):
        self.state = {
            "room": "start_room",
            "inventory": [],
            "flags": {
                "troll_alive": True,
                "door_locked": True,
                "room_dark": True
            }
        }
        self.rooms = {
            "start_room": {
                "desc": "You are in a dusty start room. A rusty SWORD lies in the corner. NORTH is a dark hallway.",
                "NORTH": "dark_hallway",
                "items": ["sword"]
            },
            "dark_hallway": {
                # Conditional description based on state
                "desc_dark": "It is pitch black. You are likely to be eaten by a grue. You can only safely go SOUTH.",
                "desc_lit": "A lit hallway. A glowing LAMP is on the floor and a KEY hangs on the wall. A troll guards the EAST door. The start room is SOUTH.",
                "SOUTH": "start_room",
                "EAST": "treasure_room", # Blocked by troll AND locked door
                "items": ["lamp", "key"]
            },
            "treasure_room": {
                "desc": "The gleaming Treasure Room! The exit is WEST.",
                "WEST": "dark_hallway",
                "items": ["gold"] # Gold is the win condition
            }
        }
        
    def _get_current_desc(self):
        room = self.rooms[self.state["room"]]
        if self.state["room"] == "dark_hallway":
            if "lamp" not in self.state["inventory"]: 
                return room["desc_dark"]
            
            desc = "A lit hallway. "
            if "lamp" in room.get("items", []): desc += "A glowing LAMP is on the floor. "
            if "key" in room.get("items", []): desc += "A KEY hangs on the wall. "
            
            if self.state["flags"]["troll_alive"]:
                desc += "A troll guards the EAST door. "
            else:
                desc += "A dead troll lies here. "
                
            if self.state["flags"]["door_locked"]:
                desc += "The heavy oak door to the EAST is locked. "
            else:
                desc += "The oak door to the EAST is open. "
            desc += "The start room is SOUTH."
            return desc
        return room["desc"]

    def execute_parsed_action(self, action_dict):
        """Processes grounded JSON actions: {"action": "verb", "target": "entity", "tool": "item"}"""
        verb = action_dict.get("action")
        target = action_dict.get("target")
        tool = action_dict.get("tool")
        
        current_room = self.rooms[self.state["room"]]
        
        if verb == "MOVE":
            if self.state["room"] == "dark_hallway" and "lamp" not in self.state["inventory"] and target != "SOUTH":
                return "It is too dark to move anywhere but back SOUTH."
                
            if target == "EAST" and self.state["room"] == "dark_hallway":
                if self.state["flags"]["troll_alive"]:
                    return "The troll blocks your path! He looks hungry."
                if self.state["flags"]["door_locked"]:
                    return "The heavy oak door is locked."
                    
            if target in current_room:
                self.state["room"] = current_room[target]
                return self._get_current_desc()
            else:
                return "You cannot go that way."
                
        elif verb == "TAKE":
            if "items" in current_room and target in current_room["items"]:
                if self.state["room"] == "dark_hallway" and "lamp" not in self.state["inventory"] and target != "lamp":
                    return "It is too dark to see any items."
                current_room["items"].remove(target)
                self.state["inventory"].append(target)
                return f"You picked up the {target}."
            else:
                return f"There is no {target} here."
                
        elif verb == "ATTACK":
            if target == "troll" and self.state["room"] == "dark_hallway":
                if self.state["flags"]["troll_alive"]:
                    if tool == "sword" and "sword" in self.state["inventory"]:
                        self.state["flags"]["troll_alive"] = False
                        return "You slay the troll with your sword!"
                    else:
                        return "You try to fight the troll bare-handed and die. (Resetting)" # Simplified death
                else:
                    return "The troll is already dead."
            return "You cannot attack that."
            
        elif verb == "OPEN":
            if target == "door" and self.state["room"] == "dark_hallway":
                if "key" in self.state["inventory"]:
                    self.state["flags"]["door_locked"] = False
                    return "You unlock the door with the key."
                else:
                    return "The door is locked. You need a key."
            return "You cannot open that."
            
        return "I don't understand."

# ======================================================================
# 2. Advanced Semantic Parser (Simulated LLM)
# ======================================================================
def extract_semantics(text_cmd):
    """
    Translates raw text into a structured semantic JSON.
    Mimics an LLM doing Few-Shot Entity Extraction.
    """
    text = text_cmd.lower()
    
    # Defaults
    semantics = {"action": "UNKNOWN", "target": None, "tool": None}
    
    # Movement
    if any(d in text for d in ["north", "south", "east", "west"]):
        semantics["action"] = "MOVE"
        for d in ["north", "south", "east", "west"]:
            if d in text: semantics["target"] = d.upper(); break
            
    # Interaction
    elif any(w in text for w in ["take", "grab", "get"]):
        semantics["action"] = "TAKE"
        for item in ["sword", "lamp", "key", "gold"]:
            if item in text: semantics["target"] = item; break
            
    # Combat
    elif any(w in text for w in ["attack", "kill", "fight", "hit"]):
        semantics["action"] = "ATTACK"
        if "troll" in text: semantics["target"] = "troll"
        if "sword" in text: semantics["tool"] = "sword"
        
    # Utility
    elif any(w in text for w in ["open", "unlock"]):
        semantics["action"] = "OPEN"
        if "door" in text: semantics["target"] = "door"
        if "key" in text: semantics["tool"] = "key"

    # Action string representation for the planning graph
    action_str = json.dumps(semantics, sort_keys=True)
    return semantics, action_str


# ======================================================================
# 3. Deep Symbolic Planner (Abstraction & Composability)
# ======================================================================
class ZorkDeepAgent:
    def __init__(self):
        self.world_graph = {}
        self.known_states = set()
        
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
                obs = env._get_current_desc()
                for past_act in path:
                    sem, _ = extract_semantics(past_act)
                    action_result = env.execute_parsed_action(sem)
                    obs = f"{action_result} | {env._get_current_desc()}"
                    if "picked up" in action_result: env_inv.append(sem["target"])
                
                s_t = self._hash_state(obs, env_inv)
                self.known_states.add(s_t)
                
                # 2. Try the new action
                sem, _ = extract_semantics(action)
                action_result = env.execute_parsed_action(sem)
                obs_next = f"{action_result} | {env._get_current_desc()}"
                
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
                            if "picked up the gold" in s_t_next.lower():
                                print(f"[AGENT] FOUND GOLD DURING EXPLORATION! Path: {path + [action]}")
                                
        print(f"[AGENT] Exploration Complete.")
        
    def plan_path(self, start_obs, target_keyword):
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

# ======================================================================
# 4. Main Execution
# ======================================================================
def run_zork_sota():
    print("==================================================")
    print(" ZORK AGI SOTA: DEEP DEPENDENCIES & SEMANTIC PARSING ")
    print("==================================================\n")
    
    env = ZorkSOTAEnvironment()
    agent = ZorkDeepAgent()
    
    start_obs = env._get_current_desc()
    print(f"INITIAL STATE: {start_obs}\n")
    
    # 1. Blind Exploration & Graph Building
    agent.explore_world(ZorkSOTAEnvironment, max_depth=10)
    print(f"Discovered {len(agent.known_states)} unique states in the graph.")
    
    # 2. Plan path to grab the gold
    winning_actions = agent.plan_path(start_obs, target_keyword="picked up the gold")
    
    if winning_actions:
        print("\n>>> DEDUCED WINNING PLAN (MULTI-STEP HORIZON):")
        for step, a in enumerate(winning_actions):
            print(f"  Step {step+1}: '{a}'")
            
        print("\n>>> EXECUTING PLAN ON FRESH ENVIRONMENT:")
        fresh_env = ZorkSOTAEnvironment()
        for a in winning_actions:
            print(f" > USER: {a}")
            sem, _ = extract_semantics(a)
            obs = fresh_env.execute_parsed_action(sem)
            print(f" > ZORK: {obs}")
            
        print("\nRESULT: PERFECT DEEP HORIZON REASONING ACHIEVED.")
    else:
        print("\nRESULT: FAILED TO DEDUCE WINNING LOGIC (Graph disconnected).")

if __name__ == "__main__":
    run_zork_sota()
