"""
V2: Larger population, diverse mutations, tighter timeout,
    with domain-aware lens optimisation context.
"""
from math import lcm
import os
import sys

# Ensure the blade-framework root is on sys.path so that framework-level
# modules (contextual_lens_problem, config, …) are importable from this subdirectory.
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
    "name": "Lens V3",
    "description": "LLaMEA with large population and domain-aware mutations (context)",
    "context": True,
    "version": "v3",
}


def configure_run(llm, n_jobs):
    """
    Return a fully configured Experiment ready to be called.

    Args:
        llm: Pre-initialized LLM instance from config.
        n_jobs: Worker count from config.

    Returns:
        iohblade.experiment.Experiment instance.
    """
    budget = 100

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable, black-box optimization.\n\n"
        "### Problem Physics & Landscape:\n"
        "Your task is to MINIMIZE a 24-dimensional camera lens design loss function with strictly bounded `[-1, 1]` parameters.\n"
        "- Indices `x[0:18]`: 18 Continuous parameters (lens curvatures and distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs (evaluated as continuous `[-1, 1]` under the hood).\n"
        "The landscape is highly non-convex and filled with infeasible 'cliffs' where invalid lenses return extremely high loss (`inf`).\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for best solution and `es.result[1]` for best fitness. To get population size, use `es.popsize` (NOT population_size). Remember `es.ask()` returns a LIST of arrays.\n"
        "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`.\n"
        "3. LHS SAMPLING: When calling the Latin Hypercube sampler, you MUST use keyword arguments to avoid shape mix-ups: `samples = lhs(n_samples=20, n_dim=self.dim)`.\n"
        "4. SCOPING: Define all logic within the `Optimizer` class. If you use helper methods, they MUST accept `func` and `grad_func` as arguments explicitly.\n\n"
        "You must track your function evaluations and strictly adhere to `self.budget`."
    )

    example_prompt = (
        "Write a completely self-contained Python class named exactly `Optimizer`.\n"
        "```python\n"
        "import numpy as np\n"
        "import cma\n"
        "from scipy.optimize import differential_evolution\n\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n\n"
        "    def __call__(self, func, grad_func=None) -> tuple[float, np.ndarray]:\n"
        "        best_f = float('inf')\n"
        "        best_x = np.zeros(self.dim)\n"
        "        \n"
        "        # ALWAYS use keyword arguments for lhs:\n"
        "        initial_population = lhs(n_samples=10, n_dim=self.dim)\n"
        "        \n"
        "        # Track your budget!\n"
        "        evals = 0\n"
        "        for x in initial_population:\n"
        "            if evals >= self.budget: break\n"
        "            f = func(x)\n"
        "            evals += 1\n"
        "            if f < best_f:\n"
        "                best_f = f\n"
        "                best_x = x\n\n"
        "        # Apply your advanced mixed-variable strategy here...\n"
        "                \n"
        "        return best_f, best_x\n"
        "```\n\n"
    )



    mutation_prompts = [
        # Strategy 1: Hybridization (Global -> Local)
        "Implement a two-phase optimization strategy. Use a robust global explorer "
        "(like Differential Evolution) for the first 80% of the budget to find "
        "promising regions, then switch to a high-precision local search (like "
        "CMA-ES or Nelder-Mead) for final refinement.",
        
        # Strategy 2: Specialized Discrete Handling
        "Dimensions 18-23 are categorical glass material IDs. Refine the algorithm "
        "to treat these dimensions as discrete integers using categorical crossover "
        "or rounding, while maintaining high-precision continuous search for the "
        "first 18 dimensions (curvatures and distances).",
        
        # Strategy 3: Stagnation & Restarts
        "The lens design landscape is highly multimodal. Implement a multi-start or "
        "restart strategy that detects when the search has stalled in a local "
        "optimum and re-initializes in a new region while preserving the best "
        "solution found so far.",
        
        # Strategy 4: Adaptive Parameters
        "Improve the algorithm by making its internal parameters (like step size, "
        "mutation factor F, or crossover rate CR) self-adaptive, so they evolve "
        "during the run based on the success of the optimization.",
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v3",
        n_parents=6,
        n_offspring=12,
        elitism=True,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 10)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=5000,
        eval_timeout=600,
        name="DoubleGauss_v3",
        task_prompt = task_prompt,
        example_prompt = example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3")

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

    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()
