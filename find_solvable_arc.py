import glob, json, time, sys
from domains.arc.env import Grid, ARCEnvironment, load_official_arc_task
from domains.arc.search import ARCBeamSearch

files = glob.glob("data/ARC-AGI/data/training/*.json")
solvable = []

print("Scanning for solvable ARC tasks...")
for i, f in enumerate(files[:100]): # Just scan first 100 for speed
    train_ex, test_ex = load_official_arc_task(f)
    agent = ARCBeamSearch()
    # Tiny beam for speed
    import os
    sys.stdout = open(os.devnull, 'w')
    prog = agent.search(train_ex, target=None, beam_width=5, max_generations=5)
    sys.stdout = sys.__stdout__
    
    if prog:
        # Check test
        env = ARCEnvironment(test_ex[0][0])
        pred = env.execute_action(prog)
        if pred == test_ex[0][1]:
            print(f"[*] SOLVED: {f} using {prog}")
            solvable.append(f)

print("Total solvable found:", len(solvable))
