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
import cma

class Optimizer:
    def __init__(self, budget, dim, grad0_cont):
        self.budget = budget
        self.dim = dim
        self.grad0_cont = grad0_cont

    def __call__(self, func, grad_func=None):
        best_f = float('inf')
        best_x = None
        
        # FIX 1: Explicitly pass arguments to prevent shape mixing
        initial_population = lhs(n_samples=10, n_dim=self.dim)
        
        # Bias the initial population distribution based on the gradient information
        for i in range(initial_population.shape[0]):
            initial_population[i, :18] += 0.05 * self.grad0_cont
        
        # Initialize CMA-ES with the biased initial population
        es = cma.CMAEvolutionStrategy(initial_population.mean(axis=0), 0.3)
        
        # Track evaluations to respect the budget
        evals = 0
        
        while evals < self.budget:
            # FIX 2: es.ask() returns a full list of solutions (the population)
            solutions = es.ask()
            fitness_values = []
            
            for x in solutions:
                # Stop if we hit the strict evaluation budget
                if evals >= self.budget:
                    # Provide a dummy high fitness for unevaluated samples just to keep CMA-ES happy
                    fitness_values.append(float('inf'))
                    continue
                    
                f = func(x)
                evals += 1
                
                if f < best_f:
                    best_f = f
                    best_x = x
                    
                fitness_values.append(f)
            
            # Tell CMA-ES the results of the evaluations
            es.tell(solutions, fitness_values)
            
            # Optional: Stop early if CMA-ES converges
            if es.stop():
                break
        
        return best_f, best_x
"""

sol = Solution(code=code.strip())
prob = LensOptimisation(budget_factor=50, training_instances=[(1,)])






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
