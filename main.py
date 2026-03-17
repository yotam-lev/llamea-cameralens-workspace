#!/usr/bin/env python3
"""
Main entry point for the LLaMEA Camera Lens Workspace.
This script provides an interactive CLI to select and run experiments.
"""

import sys
import os
import argparse
from pathlib import Path

# Identify the project root and blade-framework directory
_PROJECT_ROOT = Path(__file__).parent.resolve()
_BLADE_DIR = _PROJECT_ROOT / "blade-framework"

def smoke_test():
    """Run a quick single evaluation to verify the environment."""
    print("\n" + "="*60)
    print("SMOKE TEST: Running a single evaluation (1 seed, budget 5000)")
    print("="*60)
    
    if str(_BLADE_DIR) not in sys.path:
        sys.path.insert(0, str(_BLADE_DIR))
    
    os.chdir(_BLADE_DIR)
    
    try:
        from iohblade.problems.lens_optimisation import LensOptimisation
        from iohblade.solution import Solution
        import time
        import numpy as np
        
        problem = LensOptimisation(training_instances=[(1,)])
        code = """
import numpy as np
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    def __call__(self, func):
        best_f = float('inf')
        best_x = None
        for _ in range(self.budget):
            x = np.random.uniform(-1, 1, self.dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        return best_f, best_x
"""
        sol = Solution(code=code)
        
        start = time.time()
        print("Evaluating...")
        result = problem.evaluate(sol)
        end = time.time()
        
        print(f"\nResult Fitness: {result.fitness}")
        print(f"Feedback: {result.feedback}")
        if result.error:
            print(f"Error: {result.error}")
        
        print(f"\nSmoke test completed in {end-start:.2f} seconds.")
        if result.fitness > -np.inf:
            print("✅ SUCCESS: Environment is working correctly.")
        else:
            print("❌ FAILURE: Evaluation returned -inf.")
            
    except Exception as e:
        print(f"❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_experiments():
    """
    Sets up the environment and launches the interactive experiment runner.
    """
    parser = argparse.ArgumentParser(description="LLaMEA Camera Lens Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick verification evaluation")
    parser.add_argument("--run", help="Run a specific experiment (e.g., lens_v2)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    
    args, unknown = parser.parse_known_args()
    
    if args.smoke_test:
        smoke_test()
        return

    # Ensure blade-framework is in sys.path so its modules can be imported
    if str(_BLADE_DIR) not in sys.path:
        sys.path.insert(0, str(_BLADE_DIR))

    # Change to the blade-framework directory to ensure relative paths
    # (like results/ or local data files) are resolved correctly.
    os.chdir(_BLADE_DIR)

    try:
        from cameralens_main import main as cli_main
        # Re-inject sys.argv for the wrapped CLI
        # We only want to pass arguments that cameralens_main.py understands
        new_argv = [sys.argv[0]]
        if args.run:
            new_argv.extend(["--run", args.run])
        if args.list:
            new_argv.append("--list")
        
        # Also pass through any unknown arguments that might be intended for the underlying CLI
        # but filter out things already handled
        for arg in unknown:
            if arg not in ["--run", "--list", "--smoke-test"]:
                new_argv.append(arg)
                
        sys.argv = new_argv
        cli_main()
    except ImportError as e:
        print(f"Error: Could not import 'cameralens_main' from blade-framework.")
        print(f"Detailed error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_experiments()
