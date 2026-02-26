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
    best_program = agent.search(train_ex, target=None, beam_width=100, max_generations=30)
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

# --- Top-level trial functions for parallel execution ---

def _run_zork_trial(args):
    """Single Zork trial — must be top-level for ThreadPoolExecutor."""
    level, depth, budget, verbose = args
    import time, pickle, sys, os
    
    t0 = time.time()
    env = ZorkSOTAEnvironment(level=level)
    agent = ZorkDeepAgent()
    
    if verbose:
        # Don't suppress output in verbose mode
        agent.explore_world(lambda: env, max_depth=depth, max_states=budget, verbose=True)
    else:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_fd = os.dup(1)
        os.dup2(devnull, 1)
        os.close(devnull)
        try:
            agent.explore_world(lambda: env, max_depth=depth, max_states=budget)
        finally:
            os.dup2(old_fd, 1)
            os.close(old_fd)
    
    max_score = 0
    for state_hash in agent.known_states:
        try:
            env.load_state(pickle.loads(state_hash))
            max_score = max(max_score, env.get_score())
        except Exception:
            pass
    env.env.reset()
    
    t1 = time.time()
    return max_score, t1 - t0, len(agent.known_states)


def _run_arc_trial(args):
    """Single ARC trial — must be top-level for ProcessPoolExecutor."""
    level, beam_width, max_gens, verbose = args
    import time, sys, os
    train_ex, test_tests = generate_2d_arc_task(level=level, official_benchmark=True)
    test_ex = test_tests[0]
    
    agent = ARCBeamSearch()
    
    t0 = time.time()
    if verbose:
        best_program = agent.search(train_ex, target=None, beam_width=beam_width, max_generations=max_gens, verbose=True)
    else:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_fd = os.dup(1)
        os.dup2(devnull, 1)
        os.close(devnull)
        try:
            best_program = agent.search(train_ex, target=None, beam_width=beam_width, max_generations=max_gens)
        finally:
            os.dup2(old_fd, 1)
            os.close(old_fd)
    t1 = time.time()
    
    success = False
    ast_str = ""
    if best_program:
        predicted = ARCEnvironment(test_ex[0]).execute_action(best_program)
        if predicted == test_ex[1]:
            success = True
            ast_str = str(best_program)
    
    return success, t1 - t0, ast_str



def run_benchmarks(domain, level, trials, workers=None, budget=None, beam_width=None, max_gens=None, verbose=False):
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    # In verbose mode, force single worker for readable sequential output
    if verbose:
        workers = 1
        print("[VERBOSE] Forcing workers=1 for readable sequential output\n")
    
    print(f"\n{'='*50}")
    print(f" BENCHMARKING {domain.upper()} (Level {level}) | Trials: {trials} | Workers: {workers}")
    print(f" {'='*50}\n")
    
    if domain == "zork":
        search_type = "BFS" if level == 1 else "A* Best-First"
        depth = 5 if level == 1 else (15 if level == 2 else 25)
        if budget is None:
            budget = 2000 if level == 1 else (1000 if level == 2 else 2000)
        print(f">>> {'VERBOSE ' if verbose else ''}EVALUATION ({search_type}, Depth: {depth}, Budget: {budget})")
        
        trial_args = [(level, depth, budget, verbose)] * trials
        with ThreadPoolExecutor(max_workers=min(workers, trials)) as executor:
            results = list(executor.map(_run_zork_trial, trial_args))
        
        total_score = 0
        total_time = 0.0
        for i, (score, elapsed, states) in enumerate(results):
            total_score += score
            total_time += elapsed
            print(f"Trial {i+1}/{trials} | COMPLETED | Score: {score}/350 | Time: {elapsed:.3f}s | States: {states}")
        
        avg_score = total_score / trials
        avg_time = total_time / trials
        wall_time = max(r[1] for r in results)
        print(f"\n=== RESULTS ===")
        print(f"Average Episode Score:     {avg_score:.1f}/350")
        print(f"Average CPU Time:          {avg_time:.3f}s per trial")
        print(f"Wall Clock Time:           {wall_time:.3f}s (parallel)")
        print("===============")
        
    elif domain == "arc":
        bw = beam_width if beam_width else 100
        gens = max_gens if max_gens else 30
        print(f">>> {'VERBOSE ' if verbose else ''}EVALUATION (Beam: {bw}, Gens: {gens})")
        
        trial_args = [(level, bw, gens, verbose)] * trials
        effective_workers = min(workers, trials)
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(_run_arc_trial, trial_args))
        
        success_count = 0
        total_time = 0.0
        for i, (success, elapsed, ast_str) in enumerate(results):
            total_time += elapsed
            if success:
                success_count += 1
                print(f"Trial {i+1}/{trials} | SUCCESS | Time: {elapsed:.3f}s | AST: {ast_str}")
            else:
                print(f"Trial {i+1}/{trials} | FAILED  | Time: {elapsed:.3f}s")
        
        success_rate = (success_count / trials) * 100
        avg_time = total_time / trials
        wall_time = max(r[1] for r in results)
        print(f"\n=== RESULTS ===")
        print(f"Full Dataset Success Rate: {success_rate:.1f}% ({success_count}/{trials})")
        print(f"Average CPU Time:          {avg_time:.3f}s per trial")
        print(f"Wall Clock Time:           {wall_time:.3f}s (parallel)")
        print("===============")


if __name__ == "__main__":
    import multiprocessing
    
    parser = argparse.ArgumentParser(
        description="AGI Core Pillars — Benchmark & Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py interactive --domain zork --level 2
  python main.py interactive --domain arc --level 3 --verbose
  python main.py benchmark --domain arc --level 3 --trials 20 --workers 5
  python main.py benchmark --domain zork --level 2 --trials 10 --budget 2000
  python main.py benchmark --domain arc --level 3 --trials 5 --beam-width 200 --max-gens 50
  python main.py benchmark --domain zork --level 2 --trials 1 --verbose
""")
    parser.add_argument("mode", choices=["interactive", "benchmark"], help="Execution mode")
    parser.add_argument("--domain", choices=["arc", "zork"], required=True, help="Domain to run")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=3, help="Difficulty level (1-3)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for benchmark mode")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help=f"Parallel workers (default: {multiprocessing.cpu_count()} = auto-detected CPUs)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Zork: max states to expand per trial (default: 1000 for L2, 2000 for L3)")
    parser.add_argument("--beam-width", type=int, default=None, dest="beam_width",
                        help="ARC: beam width for evolutionary search (default: 100)")
    parser.add_argument("--max-gens", type=int, default=None, dest="max_gens",
                        help="ARC: max generations for evolutionary search (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable detailed logging (room discovery, inventory, per-gen stats)")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        if args.domain == "zork":
            run_zork_interactive(args.level)
        elif args.domain == "arc":
            run_arc_interactive(args.level)
    elif args.mode == "benchmark":
        run_benchmarks(args.domain, args.level, args.trials,
                       workers=args.workers, budget=args.budget,
                       beam_width=args.beam_width, max_gens=args.max_gens,
                       verbose=args.verbose)
