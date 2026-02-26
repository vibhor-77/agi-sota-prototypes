import argparse
import time

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
    # Dynamic depth based on level (1: shallow, 3: deep)
    depth = {1: 4, 2: 6, 3: 10}[level]
    
    t0 = time.time()
    agent.explore_world(lambda: ZorkSOTAEnvironment(level=level), max_depth=depth)
    t1 = time.time()
    
    print(f"[*] Discovered {len(agent.known_states)} unique states in {t1-t0:.2f}s.\n")
    
    print("[AGENT] Planning path to 'picked up the gold'...")
    winning_actions = agent.search(start_obs, target_keyword="picked up the gold")
    
    if winning_actions:
        print("\n>>> DEDUCED WINNING PLAN:")
        for step, a in enumerate(winning_actions):
            print(f"  Step {step+1}: '{a}'")
            
        print("\n>>> EXECUTING PLAN ON FRESH ENVIRONMENT:")
        fresh_env = ZorkSOTAEnvironment(level=level)
        for a in winning_actions:
            print(f" > USER: {a}")
            sem = parser.parse(a)
            obs = fresh_env.execute_action(sem)
            print(f" > ZORK: {obs}")
            time.sleep(0.5) # Slight pause for interactive feel
            
        print(f"\n[+] RESULT: PERFECT DEEP HORIZON REASONING ACHIEVED (Level {level}).")
    else:
        print("\n[-] RESULT: FAILED TO DEDUCE WINNING LOGIC.")

def run_arc_interactive(level):
    print(f"\n{'='*50}")
    print(f" ARC AGI: LEVEL {level} INTERACTIVE DEMO")
    print(f" {'='*50}\n")
    
    examples = generate_2d_arc_task(4, level=level)
    train_ex = examples[:3]
    test_ex = examples[3]
    
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
        depth = {1: 4, 2: 6, 3: 10}[level]
        for i in range(trials):
            t0 = time.time()
            env = ZorkSOTAEnvironment(level=level)
            agent = ZorkDeepAgent()
            # Suppress prints for pure benching
            import sys, os
            sys.stdout = open(os.devnull, 'w')
            agent.explore_world(lambda: ZorkSOTAEnvironment(level=level), max_depth=depth)
            plan = agent.search(env.get_observation(), target_keyword="picked up the gold")
            sys.stdout = sys.__stdout__
            
            t1 = time.time()
            total_time += (t1 - t0)
            
            if plan:
                success_count += 1
                print(f"Trial {i+1}/{trials} | SUCCESS | Time: {t1-t0:.3f}s | Path Depth: {len(plan)}")
            else:
                print(f"Trial {i+1}/{trials} | FAILED  | Time: {t1-t0:.3f}s")
                
    elif domain == "arc":
        for i in range(trials):
            examples = generate_2d_arc_task(4, level=level)
            train_ex = examples[:3]
            test_ex = examples[3]
            
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
            
    success_rate = (success_count / trials) * 100
    avg_time = total_time / trials
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Overall Success Rate: {success_rate:.1f}% ({success_count}/{trials})")
    print(f"Average CPU Time:     {avg_time:.3f}s per trial")
    print("=========================")


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
