import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.zork.semantics import ZorkSemanticParser

def main():
    print("\n==============================================")
    print(" 🕹️ Zork NLP Semantic Graph Example 🕹️")
    print("==============================================\n")

    parser = ZorkSemanticParser()

    print("[1] The agent attempts to map raw linguistic states into a logic graph.")
    print("    However, raw text suffers from 'aliasing'—different phrases mean ")
    print("    the same entity. We extract clean 'Semantics' to fix this.\n")

    sentences = [
        "I wander aimlessly and eventually go north",
        "please quickly grab the shiny sword",
        "i bravely attack the nasty troll using my rusty sword",
        "unlock the heavy oak door with the shiny brass key"
    ]

    for i, s in enumerate(sentences):
        print(f"--- Example {i+1} ---")
        print(f"    Raw Player Input : \"{s}\"")
        parsed = parser.parse(s)
        print(f" -> Parsed Semantics : {parsed}\n")

    print("🚀 INSIGHT:")
    print("By mapping messy text into a clean logic trace, the A* Search algorithm")
    print("can navigate thousands of alternate-timeline universes without crashing into")
    print("the deterministic loops that traditionally destroy deep symbolic agents!")
    print("==============================================\n")

if __name__ == "__main__":
    main()
