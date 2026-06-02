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
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.grad0_cont = None

    def set_initial_gradient(self, grad0_cont):
        self.grad0_cont = grad0_cont

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None):
        # Initialization (LHS)
        initial_population = lhs(n_samples=20, n_dim=self.dim)
        for x in initial_population:
            self._evaluate(x, func)

        # Initial gradient step
        if grad_func is not None and self.grad0_cont is not None:
            baseline_x = np.zeros(self.dim)
            lr = 0.01  # Learning rate for the gradient step
            gradient_step = lr * self.grad0_cont[:18]
            baseline_x[:18] -= gradient_step
            self._evaluate(baseline_x, func)

        # Enhanced Random Search with Gaussian mutation 
        while self.evals < self.budget:
            if np.random.rand() < 0.5:
                x = np.random.uniform(-1, 1, self.dim)
            else:
                # Gaussian mutation around the best found solution
                sigma = 0.1  # Mutation strength
                x = self.best_x.copy()
                x[:18] += np.random.normal(0, sigma, 18)  # Mutate continuous variables
                # Ensure bounds are respected for continuous variables
                x[:18] = np.clip(x[:18], -1, 1)
                # Categorical variables remain unchanged as they are discrete

            self._evaluate(x, func)

        return self.best_f, self.best_x
  
"""

sol = Solution(code=code.strip())
prob = LensOptimisation(budget_factor=10000, training_instances=[(1,3)])






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
print(f"this is the evaluated solution{evaluated_sol}")

print(f"Fitness: {evaluated_sol.fitness}")
print(f"Feedback: {evaluated_sol.feedback}")
print(f"Error: {evaluated_sol.error}")
