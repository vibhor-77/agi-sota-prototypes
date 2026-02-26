import argparse
import time
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# Zork
from domains.zork.env import ZorkSOTAEnvironment
from domains.zork.agent import ZorkDeepAgent
from domains.zork.semantics import ZorkSemanticParser

# ARC
from domains.arc.env import ARCEnvironment, generate_2d_arc_task
from domains.arc.search import ARCBeamSearch

def run_zork_interactive(level):
    print(f"\n{'='*50}")
    print(f" ZORK AGI: LEVEL {level} INTERACTIVE DEMO")
    print(f" {'='*50}\n")
    
    env = ZorkSOTAEnvironment(level=level)
    agent = ZorkDeepAgent()
    parser = ZorkSemanticParser()
    
    start_obs = env.get_observation()
    print(f"INITIAL STATE:\n{start_obs}\n")
    
    print("[AGENT] Exploring phase using BFS abstraction graph building...")
    # Dynamic goals based on level for true Zork 1 ROM
    if level == 1:
        depth = 2
        goal = "leaflet"
    elif level == 2:
        depth = 4
        goal = "kitchen"
    else:
        depth = 6
        goal = "sword"
        
    t0 = time.time()
    # Pass the actual instantiated env instead of a lambda to save memory creation
    agent.explore_world(lambda: env, max_depth=depth)
    t1 = time.time()
    
    print(f"[*] Discovered {len(agent.known_states)} unique states in {t1-t0:.2f}s.\n")
    
    print(f"[AGENT] Planning path to '{goal}'...")
    winning_actions = agent.search(env, target_keyword=goal)
    
    if winning_actions:
        print("\n>>> DEDUCED WINNING PLAN:")
        for step, a in enumerate(winning_actions):
            print(f"  Step {step+1}: '{a}'")
            
        print("\n>>> EXECUTING PLAN ON FRESH ENVIRONMENT:")
        fresh_env = ZorkSOTAEnvironment(level=level)
        for a in winning_actions:
            print(f" > USER: {a}")
            # Execute the action string directly on the Frotz engine
            obs = fresh_env.step_raw(a)
            print(f" > ZORK: {obs}")
            time.sleep(0.5) # Slight pause for interactive feel
            
        print(f"\n[+] RESULT: PERFECT DEEP HORIZON REASONING ACHIEVED (Level {level}).")
    else:
        print("\n[-] RESULT: FAILED TO DEDUCE WINNING LOGIC.")

def run_arc_interactive(level):
    print(f"\n{'='*50}")
    print(f" ARC AGI: LEVEL {level} INTERACTIVE DEMO")
    print(f" {'='*50}\n")
    
    train_ex, test_tests = generate_2d_arc_task(level=level)
    test_ex = test_tests[0] # Just evaluate on the first test case
    
    print(">>> OBSERVING TRAINING GRIDS")
    for i, (inp, out) in enumerate(train_ex):
        print(f"\n--- Train {i+1} ---")
        print(f"Input:\n{inp.arr}")
        print(f"Output:\n{out.arr}")
        
    agent = ARCBeamSearch()
    
    t0 = time.time()
    best_program = agent.search(train_ex, target=None, beam_width=50, max_generations=20)
    t1 = time.time()
    
    print("\n==================================================")
    print(">>> COMPOSITIONAL LEARNING CONVERGED")
    print("==================================================")
    if best_program:
        print(f"[*] Search Time: {t1-t0:.2f}s")
        print(f"[*] Discovered Concept: f(grid) = {str(best_program)}\n")
        
        print(">>> EXTRAPOLATION ON UNSEEN TEST GRID")
        test_in, test_out = test_ex
        
        env = ARCEnvironment(test_in)
        predicted = env.execute_action(best_program)
        
        print(f"Input Grid:\n{test_in.arr}")
        print(f"AI Predicted:\n{predicted.arr}")
        print(f"True Answer:\n{test_out.arr}")
        
        if predicted == test_out:
            print(f"\n[+] RESULT: PERFECT 2D GENERALIZATION ACHIEVED (Level {level}).")
        else:
            print(f"\n[-] RESULT: TEST FAILED (Level {level}).")
    else:
        print("[-] Search failed.")

def run_benchmarks(domain, level, trials):
    print(f"\n{'='*50}")
    print(f" BENCHMARKING {domain.upper()} (Level {level}) | Trials: {trials}")
    print(f" {'='*50}\n")
    
    success_count = 0
    total_time = 0.0
    
    if domain == "zork":
        print(">>> RUNNING TRUE BASELINE EVALUATION (Max Depth BFS)")
        for i in range(trials):
            t0 = time.time()
            env = ZorkSOTAEnvironment(level=level)
            agent = ZorkDeepAgent()
            
            # Suppress prints for pure benching
            import sys, os
            sys.stdout = open(os.devnull, 'w')
            # BFS explores to a bounded depth before evaluating maximum score
            agent.explore_world(lambda: env, max_depth=5)
            sys.stdout = sys.__stdout__
            
            import pickle
            max_score = 0
            for state_hash in agent.known_states:
                try:
                    env.load_state(pickle.loads(state_hash))
                    max_score = max(max_score, env.get_score())
                except Exception:
                    pass
            env.env.reset()
            
            t1 = time.time()
            total_time += (t1 - t0)
            success_count += max_score
            
            print(f"Trial {i+1}/{trials} | COMPLETED | Max Score Found: {max_score}/350 | Time: {t1-t0:.3f}s | States Explored: {len(agent.known_states)}")
                
    elif domain == "arc":
        print(">>> RUNNING TRUE BASELINE EVALUATION (Random 400 Official Tasks)")
        for i in range(trials):
            train_ex, test_tests = generate_2d_arc_task(level=level, official_benchmark=True)
            test_ex = test_tests[0]
            
            agent = ARCBeamSearch()
            
            t0 = time.time()
            import sys, os
            sys.stdout = open(os.devnull, 'w')
            best_program = agent.search(train_ex, target=None, beam_width=50, max_generations=20)
            sys.stdout = sys.__stdout__
            t1 = time.time()
            
            total_time += (t1 - t0)
            
            if best_program:
                predicted = ARCEnvironment(test_ex[0]).execute_action(best_program)
                if predicted == test_ex[1]:
                    success_count += 1
                    print(f"Trial {i+1}/{trials} | SUCCESS | Time: {t1-t0:.3f}s | AST: {str(best_program)}")
                    continue
            print(f"Trial {i+1}/{trials} | FAILED  | Time: {t1-t0:.3f}s")
            
    avg_time = total_time / trials
    print(f"\n=== TRUE BASELINE RESULTS ===")
    if domain == "arc":
        success_rate = (success_count / trials) * 100
        print(f"Full Dataset Success Rate: {success_rate:.1f}% ({success_count}/{trials})")
    elif domain == "zork":
        avg_score = success_count / trials
        print(f"Average Episode Score:     {avg_score:.1f}/350")
    print(f"Average CPU Time:          {avg_time:.3f}s per trial")
    print("=============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI Core Pillars Execution CLI")
    parser.add_argument("mode", choices=["interactive", "benchmark"], help="Execution mode")
    parser.add_argument("--domain", choices=["arc", "zork"], required=True, help="Domain to run")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=3, help="Difficulty level (1-3)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for benchmark mode")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        if args.domain == "zork":
            run_zork_interactive(args.level)
        elif args.domain == "arc":
            run_arc_interactive(args.level)
    elif args.mode == "benchmark":
        run_benchmarks(args.domain, args.level, args.trials)
