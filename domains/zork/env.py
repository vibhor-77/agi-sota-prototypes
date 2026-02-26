import json
from core.feedback import Environment

class ZorkSOTAEnvironment(Environment):
    """
    Pillar 1: Feedback Loop.
    A hidden deterministic state machine with multi-step dependencies.
    """
    def __init__(self, level=3):
        self.level = level
        self.state = {
            "room": "start_room",
            "inventory": [],
            "flags": {
                "troll_alive": level >= 2,
                "door_locked": level >= 1,
                "room_dark": level >= 3
            }
        }
        
        self.rooms = {}
        if level == 1:
            self.rooms = {
                "start_room": {
                    "desc": "A dusty room. The EAST door is locked. A KEY hangs on the wall.",
                    "EAST": "treasure_room",
                    "items": ["key"]
                },
                "treasure_room": {
                    "desc": "The gleaming Treasure Room! The exit is WEST.",
                    "WEST": "start_room",
                    "items": ["gold"]
                }
            }
        elif level == 2:
             self.rooms = {
                "start_room": {
                    "desc": "A dusty room. A SWORD lies here. The EAST door is open but a troll guards it.",
                    "EAST": "treasure_room",
                    "items": ["sword"]
                },
                "treasure_room": {
                    "desc": "The gleaming Treasure Room! The exit is WEST.",
                    "WEST": "start_room",
                    "items": ["gold"]
                }
            }
        else:
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
        if self.level == 3 and self.state["room"] == "dark_hallway":
            if "lamp" not in self.state["inventory"]: return room["desc_dark"]
            desc = "A lit hallway. "
            if "lamp" in room.get("items", []): desc += "A glowing LAMP is on the floor. "
            if "key" in room.get("items", []): desc += "A KEY hangs on the wall. "
            desc += "A troll guards the EAST door. " if self.state["flags"]["troll_alive"] else "A dead troll lies here. "
            desc += "The heavy oak door to the EAST is locked. " if self.state["flags"]["door_locked"] else "The oak door to the EAST is open. "
            desc += "The start room is SOUTH."
            return desc
            
        if self.level == 2 and self.state["room"] == "start_room":
            desc = "A dusty room. "
            if "sword" in room.get("items", []): desc += "A SWORD lies here. "
            desc += "A troll guards the EAST door." if self.state["flags"]["troll_alive"] else "A dead troll lies here. The EAST door is open."
            return desc
            
        if self.level == 1 and self.state["room"] == "start_room":
            desc = "A dusty room. "
            if "key" in room.get("items", []): desc += "A KEY hangs on the wall. "
            desc += "The EAST door is locked." if self.state["flags"]["door_locked"] else "The EAST door is open."
            return desc
            
        return room.get("desc", str(room))

    def execute_action(self, action_dict):
        verb = action_dict.get("action")
        target = action_dict.get("target")
        tool = action_dict.get("tool")
        current_room = self.rooms[self.state["room"]]
        
        # Determine which room holds the gate constraints
        conflict_room = "dark_hallway" if self.level == 3 else "start_room"
        
        if verb == "MOVE":
            if self.level == 3 and self.state["room"] == "dark_hallway" and "lamp" not in self.state["inventory"] and target != "SOUTH":
                return "It is too dark to move anywhere but back SOUTH."
                
            if target == "EAST" and self.state["room"] == conflict_room:
                if self.state["flags"]["troll_alive"]: return "The troll blocks your path! He looks hungry."
                if self.state["flags"]["door_locked"]: return "The heavy door is locked."
                    
            if target in current_room:
                self.state["room"] = current_room[target]
                return self.get_observation()
            else:
                return "You cannot go that way."
                
        elif verb == "TAKE":
            if "items" in current_room and target in current_room["items"]:
                if self.level == 3 and self.state["room"] == "dark_hallway" and "lamp" not in self.state["inventory"] and target != "lamp":
                    return "It is too dark to see any items."
                current_room["items"].remove(target)
                self.state["inventory"].append(target)
                return f"You picked up the {target}."
            else:
                return f"There is no {target} here."
                
        elif verb == "ATTACK":
            if target == "troll" and self.state["room"] == conflict_room and self.level >= 2:
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
            if target == "door" and self.state["room"] == conflict_room and (self.level == 1 or self.level == 3):
                if "key" in self.state["inventory"]:
                    self.state["flags"]["door_locked"] = False
                    return "You unlock the door with the key."
                else:
                    return "The door is locked. You need a key."
            return "You cannot open that."
            
        return "I don't understand."
