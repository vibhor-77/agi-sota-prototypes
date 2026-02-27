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

def run_zork_interactive(level, verbose=False):
    print(f"\n{'='*50}")
    print(f" ZORK AGI: LEVEL {level} INTERACTIVE DEMO {'(VERBOSE)' if verbose else ''}")
    print(f" {'='*50}\n")
    
    env = ZorkSOTAEnvironment(level=level)
    agent = ZorkDeepAgent()
    parser = ZorkSemanticParser()
    
    start_obs = env.get_observation()
    print(f"INITIAL STATE:\n{start_obs}\n")
    
    if verbose:
        print(f"[VERBOSE] Game score at start: {env.get_score()}")
        try:
            print(f"[VERBOSE] Starting inventory: {env.get_inventory()}")
        except Exception:
            print(f"[VERBOSE] Starting inventory: []")
    
    # Dynamic goals based on level
    if level == 1:
        depth = 2
        goal = "leaflet"
    elif level == 2:
        depth = 4
        goal = "kitchen"
    else:
        depth = 6
        goal = "sword"
    
    print(f"[AGENT] Exploring (depth={depth}, goal='{goal}')...")
        
    t0 = time.time()
    agent.explore_world(lambda: env, max_depth=depth, verbose=verbose)
    t1 = time.time()
    
    print(f"[*] Discovered {len(agent.known_states)} unique states in {t1-t0:.2f}s.\n")
    
    if verbose:
        print(f"[VERBOSE] World graph: {sum(len(v) for v in agent.world_graph.values())} edges across {len(agent.world_graph)} nodes")
        if agent.best_path:
            print(f"[VERBOSE] Best exploration path: {' → '.join(agent.best_path)}")
    
    print(f"[AGENT] Planning path to '{goal}'...")
    winning_actions = agent.search(env, target_keyword=goal)
    
    if winning_actions:
        print(f"\n>>> DEDUCED WINNING PLAN ({len(winning_actions)} steps):")
        for step, a in enumerate(winning_actions):
            print(f"  Step {step+1}: '{a}'")
            
        print("\n>>> EXECUTING PLAN ON FRESH ENVIRONMENT:")
        fresh_env = ZorkSOTAEnvironment(level=level)
        for a in winning_actions:
            print(f" > USER: {a}")
            obs = fresh_env.step_raw(a)
            print(f" > ZORK: {obs}")
            if verbose:
                print(f"   [VERBOSE] Score after action: {fresh_env.get_score()}")
            time.sleep(0.3)
            
        final_score = fresh_env.get_score()
        print(f"\n[+] RESULT: Plan executed. Final score: {final_score}/350 (Level {level}).")
    else:
        print("\n[-] RESULT: FAILED TO DEDUCE WINNING LOGIC.")
        if verbose:
            print(f"[VERBOSE] '{goal}' not found in any of {len(agent.state_texts)} explored state texts.")
            # Show closest matches
            for h, text in list(agent.state_texts.items())[:5]:
                print(f"  State: \"{text[:100]}...\"")

def run_arc_interactive(level, verbose=False):
    print(f"\n{'='*50}")
    print(f" ARC AGI: LEVEL {level} INTERACTIVE DEMO {'(VERBOSE)' if verbose else ''}")
    print(f" {'='*50}\n")
    
    train_ex, test_tests = generate_2d_arc_task(level=level)
    test_ex = test_tests[0]
    
    print(">>> OBSERVING TRAINING GRIDS")
    for i, (inp, out) in enumerate(train_ex):
        print(f"\n--- Train {i+1} ---")
        print(f"Input  ({inp.arr.shape[0]}×{inp.arr.shape[1]}):\n{inp.arr}")
        print(f"Output ({out.arr.shape[0]}×{out.arr.shape[1]}):\n{out.arr}")
        if verbose:
            import numpy as np
            diff_pixels = np.sum(inp.arr != out.arr) if inp.arr.shape == out.arr.shape else "N/A (shape change)"
            print(f"[VERBOSE] Pixel diff: {diff_pixels}")
        
    agent = ARCBeamSearch()
    
    t0 = time.time()
    best_program = agent.search(train_ex, target=None, beam_width=200, max_generations=50, verbose=verbose)
    t1 = time.time()
    
    print("\n==================================================")
    print(">>> COMPOSITIONAL LEARNING RESULT")
    print("==================================================")
    if best_program:
        print(f"[*] Search Time: {t1-t0:.2f}s")
        print(f"[*] Discovered Concept: f(grid) = {str(best_program)}\n")
        
        if verbose:
            # Show how the program performs on each training example
            from domains.arc.heuristics import PixelEditDistance
            hed = PixelEditDistance()
            print("[VERBOSE] Training example validation:")
            for i, (inp, out) in enumerate(train_ex):
                try:
                    pred = best_program.evaluate({'input_grid': inp})
                    loss = hed.evaluate(pred, out)
                    match = "✅ EXACT" if loss == 0.0 else f"❌ loss={loss:.4f}"
                    print(f"  Train {i+1}: {match}")
                    if loss > 0.0:
                        print(f"    Predicted: {pred.arr}")
                        print(f"    Expected:  {out.arr}")
                except Exception as e:
                    print(f"  Train {i+1}: ❌ CRASH: {e}")
        
        print(">>> EXTRAPOLATION ON UNSEEN TEST GRID")
        test_in, test_out = test_ex
        
        env = ARCEnvironment(test_in)
        predicted = env.execute_action(best_program)
        
        print(f"Input Grid  ({test_in.arr.shape[0]}×{test_in.arr.shape[1]}):\n{test_in.arr}")
        print(f"AI Predicted ({predicted.arr.shape[0]}×{predicted.arr.shape[1]}):\n{predicted.arr}")
        print(f"True Answer  ({test_out.arr.shape[0]}×{test_out.arr.shape[1]}):\n{test_out.arr}")
        
        if predicted == test_out:
            print(f"\n[+] RESULT: PERFECT 2D GENERALIZATION ACHIEVED (Level {level}).")
        else:
            print(f"\n[-] RESULT: TEST FAILED (Level {level}).")
            if verbose:
                import numpy as np
                if predicted.arr.shape == test_out.arr.shape:
                    diff = predicted.arr != test_out.arr
                    wrong = np.sum(diff)
                    total = test_out.arr.size
                    print(f"[VERBOSE] Pixel accuracy: {total - wrong}/{total} ({(total-wrong)/total*100:.1f}%)")
                    print(f"[VERBOSE] Wrong pixels at positions: {list(zip(*np.where(diff)))[:10]}{'...' if wrong > 10 else ''}")
                else:
                    print(f"[VERBOSE] Shape mismatch: predicted {predicted.arr.shape} vs expected {test_out.arr.shape}")
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
    ast_str = str(best_program) if best_program else ""
    final_loss = -1.0
    diag = ""
    
    if best_program:
        predicted = ARCEnvironment(test_ex[0]).execute_action(best_program)
        if predicted == test_ex[1]:
            success = True
            final_loss = 0.0
        else:
            # Compute diagnostic info
            import numpy as np
            from domains.arc.heuristics import PixelEditDistance
            final_loss = PixelEditDistance().evaluate(predicted, test_ex[1])
            if predicted.arr.shape == test_ex[1].arr.shape:
                wrong = int(np.sum(predicted.arr != test_ex[1].arr))
                total = test_ex[1].arr.size
                diag = f"loss={final_loss:.4f}, {wrong}/{total} wrong pixels"
            else:
                diag = f"loss={final_loss:.4f}, shape {predicted.arr.shape} vs {test_ex[1].arr.shape}"
    
    return success, t1 - t0, ast_str, final_loss, diag



def run_benchmarks(domain, level, trials, workers=None, budget=None, beam_width=None, max_gens=None, verbose=False):
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    print(f"\n{'='*50}")
    print(f" BENCHMARKING {domain.upper()} (Level {level}) | Trials: {trials} | Workers: {workers}")
    print(f" {'='*50}\n")
    
    if domain == "zork":
        search_type = "BFS" if level == 1 else "A* Best-First"
        depth = 5 if level == 1 else (15 if level == 2 else 25)
        if budget is None:
            budget = 2000 if level == 1 else (3000 if level == 2 else 5000)
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
        bw = beam_width if beam_width is not None else 200
        gens = max_gens if max_gens is not None else 50
        print(f">>> {'VERBOSE ' if verbose else ''}EVALUATION (Beam: {bw}, Gens: {gens})")
        
        trial_args = [(level, bw, gens, verbose)] * trials
        effective_workers = min(workers, trials)
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(_run_arc_trial, trial_args))
        
        success_count = 0
        total_time = 0.0
        for i, (success, elapsed, ast_str, final_loss, diag) in enumerate(results):
            total_time += elapsed
            if success:
                success_count += 1
                print(f"Trial {i+1}/{trials} | SUCCESS | Time: {elapsed:.3f}s | AST: {ast_str}")
            else:
                if verbose and ast_str:
                    print(f"Trial {i+1}/{trials} | FAILED  | Time: {elapsed:.3f}s | {diag} | Best: {ast_str}")
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
                        help="Zork: max states to expand per trial (default: 3000 for L2, 5000 for L3)")
    parser.add_argument("--beam-width", type=int, default=None, dest="beam_width",
                        help="ARC: beam width for evolutionary search (default: 200)")
    parser.add_argument("--max-gens", type=int, default=None, dest="max_gens",
                        help="ARC: max generations for evolutionary search (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable detailed logging (room discovery, inventory, per-gen stats)")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        if args.domain == "zork":
            run_zork_interactive(args.level, verbose=args.verbose)
        elif args.domain == "arc":
            run_arc_interactive(args.level, verbose=args.verbose)
    elif args.mode == "benchmark":
        run_benchmarks(args.domain, args.level, args.trials,
                       workers=args.workers, budget=args.budget,
                       beam_width=args.beam_width, max_gens=args.max_gens,
                       verbose=args.verbose)
