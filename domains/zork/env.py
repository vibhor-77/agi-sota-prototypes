import json
from core.feedback import Environment

class ZorkSOTAEnvironment(Environment):
    """
    Pillar 1: Feedback Loop.
    A hidden deterministic state machine with multi-step dependencies.
    """
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
                "desc_dark": "It is pitch black. You are likely to be eaten by a grue. You can only safely go SOUTH.",
                "desc_lit": "A lit hallway. A glowing LAMP is on the floor and a KEY hangs on the wall. A troll guards the EAST door. The start room is SOUTH.",
                "SOUTH": "start_room",
                "EAST": "treasure_room",
                "items": ["lamp", "key"]
            },
            "treasure_room": {
                "desc": "The gleaming Treasure Room! The exit is WEST.",
                "WEST": "dark_hallway",
                "items": ["gold"]
            }
        }
        
    def get_observation(self):
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

    def execute_action(self, action_dict):
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
                return self.get_observation()
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
                        return "You try to fight the troll bare-handed and die. (Resetting)"
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
