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
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func, grad_func):
        # Initialize the best solution and fitness with infinity and None
        best_f = float('inf')
        best_x = None
        
        # Generate an initial population using Latin Hypercube Sampling
        initial_population = lhs(self.dim, samples=10)
        
        # Bias the initial population distribution based on the gradient information
        for i in range(initial_population.shape[0]):
            # Use the gradient to adjust the first 18 continuous variables
            initial_population[i, :18] += 0.05 * grad0_cont
        
        # Initialize CMA-ES with the biased initial population and adaptive parameters
        es = cma.CMAEvolutionStrategy(initial_population.mean(axis=0), 0.3)
        
        for _ in range(self.budget):
            solutions = []
            fitness_values = []
            
            # Generate a new batch of solutions
            while len(solutions) < 10:
                x = es.ask()
                f = func(x)
                
                if f < best_f:
                    best_f = f
                    best_x = x
                
                solutions.append(x)
                fitness_values.append(f)
            
            # Tell CMA-ES the results of the evaluations, allowing it to adapt its parameters
            es.tell(solutions, fitness_values)
        
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
print(evaluated_sol)

print(f"Fitness: {evaluated_sol.fitness}")
print(f"Feedback: {evaluated_sol.feedback}")
print(f"Error: {evaluated_sol.error}")
