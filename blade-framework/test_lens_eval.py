import os
import sys
import numpy as np

# Add framework root to path
sys.path.insert(0, os.path.abspath("."))

from iohblade.problems.lens_optimisation import LensOptimisation
from iohblade.solution import Solution

# Mocking the objective import if needed, but it should work now with path setup
# LensOptimisation adds it in _build_objective

code = """
import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[float, np.ndarray]:
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
prob = LensOptimisation(budget_factor=10, training_instances=[(1,)])

# We need to make sure examples/double_gauss_objective.py is findable
# The LensOptimisation._build_objective does: from examples.double_gauss_objective import DoubleGaussObjective
# We found it in ../camera-lens-simulation/examples/double_gauss_objective.py
# So we need to add ../camera-lens-simulation to sys.path

CAMERA_LENS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "camera-lens-simulation")
)
if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)

print("Evaluating solution...")
evaluated_sol = prob.evaluate(sol)
print(f"Fitness: {evaluated_sol.fitness}")
print(f"Feedback: {evaluated_sol.feedback}")
print(f"Error: {evaluated_sol.error}")
