
import os
import sys
from pathlib import Path

# Setup paths to find the framework
PROJECT_ROOT = Path(__file__).parent.resolve()
BLADE_DIR = PROJECT_ROOT / "blade-framework"

if str(BLADE_DIR) not in sys.path:
    sys.path.insert(0, str(BLADE_DIR))

# Change to blade-framework to ensure relative paths for results/logs work
os.chdir(BLADE_DIR)

if __name__ == "__main__":
    try:
        from iohblade.problems.lens_optimisation import LensOptimisation
        from iohblade.solution import Solution
        import numpy as np

        print("--- Initializing Smoke Test V2 (Full Environment Emulation) ---")

        # This code will be executed INSIDE the subprocess venv
        # It MUST be a single class definition that handles its own logic
        diagnostic_code = """
import importlib
import sys
import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func, grad_func=None):
        results = []
        modules_to_test = [
            "numpy", "scipy", "jax", "cma", "lensgopt", 
            "scipy.optimize", "scipy.stats.qmc"
        ]
        
        for mod in modules_to_test:
            try:
                importlib.import_module(mod)
                results.append(f"✅ {mod}")
            except ImportError as e:
                results.append(f"❌ {mod} ({e})")
        
        # --- GRADIENT TEST ---
        try:
            from examples.double_gauss_objective import DoubleGaussObjective
            obj = DoubleGaussObjective(enable_grad=True)
            x0_cont, x0_ids = obj.init_from_templates()
            grad = obj.gradient_cont_int(x0_cont, x0_ids)
            results.append(f"✅ Gradient Computation (Norm: {np.linalg.norm(grad):.6f})")
        except Exception as e:
            results.append(f"❌ Gradient Computation ({e})")
            
        # Test if the local simulation package is reachable
        try:
            from examples.double_gauss_objective import DoubleGaussObjective
            results.append("✅ examples.double_gauss_objective")
        except ImportError as e:
            results.append(f"❌ examples.double_gauss_objective ({e})")

        feedback = " | ".join(results)
        raise ValueError(f"DIAGNOSTIC_RESULT: {feedback}")
"""

        # Create the problem instance
        problem = LensOptimisation(
            training_instances=[(1,)], 
            budget_factor=10, 
            eval_timeout=120
        )
        
        solution = Solution(code=diagnostic_code)

        print("Triggering evaluation in isolated subprocess...")
        evaluated_sol = problem(solution)

        print("\n" + "="*60)
        print("VIRTUAL ENVIRONMENT REPORT")
        print("="*60)
        
        print(f"Framework Feedback:\n{evaluated_sol.feedback}")
        
        if evaluated_sol.error:
            # Clean up the feedback if it contains the long diagnostic string
            msg = str(evaluated_sol.error)
            if "DIAGNOSTIC_RESULT:" in msg:
                diag = msg.split("DIAGNOSTIC_RESULT:")[1].split("'")[0].split('"')[0]
                print(f"\nFinal Diagnostic Info:\n{diag.strip()}")
            else:
                print(f"\nCaught Error:\n{evaluated_sol.error}")
        print("="*60)

    except Exception as e:
        print(f"Smoke test script failed: {e}")
        import traceback
        traceback.print_exc()
