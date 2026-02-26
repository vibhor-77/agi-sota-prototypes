import json
from core.abstraction import StateRepresentation

class ZorkSemanticParser(StateRepresentation):
    """
    Pillar 3: Abstraction.
    Translates raw text into structured semantic JSON using lightweight symbolic matching.
    Designed to be minimal and dependency-free (avoids heavy NLP libraries like Spacy/LLMs).
    """
    def parse(self, text_cmd: str):
        text = text_cmd.lower()
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

        return semantics
