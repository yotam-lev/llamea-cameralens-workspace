
import os
import sys
import numpy as np

# Add framework root to path
sys.path.insert(0, os.path.abspath("blade-framework"))

from iohblade.problems.lens_optimisation import LensOptimisation
from iohblade.solution import Solution

# Mocking the objective import if needed, but it should work now with path setup
# LensOptimisation adds it in _build_objective

code = """
import numpy as np
import scipy
import math
import cma
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[float, np.ndarray]:
        # Testing available modules and injected method
        print("Checking modules...")
        print("cma is in globals?", "cma" in globals())
        print("scipy is in globals?", "scipy" in globals())
        print("math is in globals?", "math" in globals())
        print("differential_evolution is in globals?", "differential_evolution" in globals())
        print("latin_hypercube_sampling is in globals?", "latin_hypercube_sampling" in globals())
        print("self.latin_hypercube_sampling is in self?", hasattr(self, "latin_hypercube_sampling"))
        
        # Test LHS
        x0 = self.latin_hypercube_sampling(1, self.dim)[0]
        f0 = func(x0)
        
        # Test DE
        bounds = [(-1, 1)] * self.dim
        res = differential_evolution(func, bounds, popsize=min(5, self.budget), maxiter=1)
        
        return res.fun, res.x
"""

# Make sure camera-lens-simulation is on sys.path for the parent process
CAMERA_LENS_ROOT = os.path.abspath(
    os.path.join(os.getcwd(), "camera-lens-simulation")
)
if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)
    os.environ["PYTHONPATH"] = CAMERA_LENS_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

sol = Solution(code=code)
prob = LensOptimisation(budget_factor=20, training_instances=[(1,)], eval_timeout=600)

print("Calling problem via __call__ (triggers subprocess with virtualenv)...")
try:
    evaluated_sol = prob(sol)
    print(f"Fitness: {evaluated_sol.fitness}")
    print(f"Feedback: {evaluated_sol.feedback}")
    if evaluated_sol.error:
        print(f"Error: {evaluated_sol.error}")
except Exception as e:
    print(f"Exception during call: {e}")
    import traceback
    traceback.print_exc()
