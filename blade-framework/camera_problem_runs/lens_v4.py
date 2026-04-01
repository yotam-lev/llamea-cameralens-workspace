"""
V4: Gradient-Aware Memetic Evolution.
Explicitly uses JAX gradients for the 18D continuous subspace.
"""
import os
import sys

# Ensure the blade-framework root is on sys.path
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs

# Metadata for the run selector
RUN_META = {
    "name": "Lens V4 (Gradient-Aware)",
    "description": "Memetic LLaMEA leveraging JAX gradients for 18D continuous subspace",
    "context": True,
    "version": "v4",
}

def configure_run(llm, n_jobs):
    budget = 20 # Evolutionary generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable, black-box optimization.\n\n"
        "### Problem Physics & Landscape:\n"
        "Your task is to MINIMIZE a 24-dimensional camera lens design loss function with strictly bounded `[-1, 1]` parameters.\n"
        "- Indices `x[0:18]`: 18 Continuous parameters (lens curvatures and distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs.\n"
        "The landscape is highly non-convex and filled with infeasible 'cliffs' where invalid lenses return extremely high loss (`inf`).\n\n"
        "### THE GRADIENT ADVANTAGE (`grad0_cont`) ###\n"
        "You are provided with `grad0_cont` (shape: `(18,)`), the exact analytical gradient of the 18 continuous parameters at the baseline design. "
        "Use this to bias your initial population or take an initial pseudo-gradient descent step. "
        "WARNING: Because `grad0_cont` is 18D and your solutions are 24D, you MUST slice your target array before applying the gradient (e.g., `x[:18] -= lr * self.grad0_cont`) to prevent NumPy broadcast crashes.\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for best solution and `es.result[1]` for best fitness. To get population size, use `es.popsize`. Remember `es.ask()` returns a LIST of arrays.\n"
        "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`.\n"
        "3. LHS SAMPLING: When calling the Latin Hypercube sampler, you MUST use keyword arguments: `samples = lhs(n_samples=20, n_dim=self.dim)`.\n"
    )

    example_prompt = (
        "Write a completely self-contained Python class named exactly `Optimizer`.\n"
        "```python\n"
        "import numpy as np\n"
        "import cma\n\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int, grad0_cont: np.ndarray):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n"
        "        # YOU MUST SAVE THE GRADIENT TO SELF\n"
        "        self.grad0_cont = grad0_cont \n\n"
        "    def __call__(self, func, grad_func=None) -> tuple[float, np.ndarray]:\n"
        "        best_f = float('inf')\n"
        "        best_x = np.zeros(self.dim)\n"
        "        \n"
        "        initial_population = lhs(n_samples=20, n_dim=self.dim)\n"
        "        \n"
        "        # Example: Biasing the continuous variables using a Gradient Descent step\n"
        "        # Notice the `-=` for MINIMIZATION and the `[:, :18]` slicing to match shapes!\n"
        "        if self.grad0_cont is not None:\n"
        "            learning_rate = 0.05\n"
        "            initial_population[:, :18] -= learning_rate * self.grad0_cont\n\n"
        "        # Ensure the bounds are respected after the gradient step\n"
        "        initial_population = np.clip(initial_population, -1, 1)\n"
        "        \n"
        "        # Track your budget!\n"
        "        evals = 0\n"
        "        # Apply your advanced mixed-variable strategy here...\n"
        "                \n"
        "        return best_f, best_x\n"
        "```\n\n"
    )

    mutation_prompts = [
        # Strategy 1: Memetic Hybrid (Global -> Local)
        "Implement a memetic strategy: Use a population-based global explorer (like DE) "
        "to search the full 24D space. For the best individuals in each generation, "
        "apply a local gradient-based 'polish' (like L-BFGS-B or SLSQP via scipy.optimize.minimize) "
        "to only the first 18 dimensions using the provided grad_func.",
        
        # Strategy 2: Gradient-Guided Mutation
        "Design a mutation operator that uses the gradient. Instead of random noise, "
        "perturb the first 18 dimensions proportionally to the negative gradient direction "
        "(-grad_func(x)) to accelerate convergence toward the local minimum.",
        
        # Strategy 3: Categorical vs. Continuous Split
        "The landscape is mixed-variable. Refine the algorithm to treat the last 6 "
        "dimensions as purely categorical (using discrete operators like crossover "
        "or rounding) while treating the first 18 as a smooth, differentiable manifold "
        "to be optimized via first-order information.",
        
        # Strategy 4: Trust-Region Refinement
        "Implement a trust-region approach: Use the global search to identify promising "
        "basins, then deploy a local search that respects the geometric bounds and uses "
        "gradients to 'descend' into the deep local optima characteristic of lens design."
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic",
        n_parents=3,
        n_offspring=12,
        elitism=False,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=5000, # MATCH PRODUCTION BUDGET
        eval_timeout=6000,   # Increased to handle 10k evals + local search
        name="DoubleGauss_v4",
        example_prompt = example_prompt,
        task_prompt = task_prompt
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4")

    return Experiment(
        methods=[llamea],
        problems=[lens_problem],
        runs=3,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=n_jobs,

    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']}")
    experiment()
