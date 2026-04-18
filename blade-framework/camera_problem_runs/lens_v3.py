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
    budget = 20  # Reduced for more focused evolutionary progress

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs (round to nearest integer inside your code).\n\n"
        "### STRICT API USAGE (DO NOT DEVIATE):\n"
        "1. LHS SAMPLING: The `lhs` function is already injected globally into your environment. DO NOT import it (e.g., NEVER write `from pyDOE import lhs`). Call it exactly like this: `samples = lhs(n_samples=N, n_dim=self.dim)`."
        "2. CMA-ES: Use exactly this: `es = cma.CMAEvolutionStrategy(x0, sigma0, inopts={'popsize': N})`. \n"
        "   Access results via `es.result[0]` (best x) and `es.result[1]` (best fitness).\n"
        "3. SCIPY MINIMIZE: `res = minimize(func, x0, method='L-BFGS-B', bounds=[(-1, 1)]*len(x0))`.\n"
        "4. BUDGET: You MUST check `if self.evals >= self.budget` before every call to `func(x)`.\n"
    )

    example_prompt = (
        "```python\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n"
        "        self.evals = 0\n"
        "        self.best_f = float('inf')\n"
        "        self.best_x = np.zeros(dim)\n"
        "\n"
        "    def _evaluate(self, x, func):\n"
        "        if self.evals >= self.budget: return float('inf')\n"
        "        f = func(x)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = x.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        # 1. Initialize with LHS\n"
        "        pop = lhs(n_samples=10, n_dim=self.dim)\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "        \n"
        "        # 2. Optimization Loop (e.g., CMA-ES for continuous 18D)\n"
        "        # Combine continuous sample x_c with best categorical best_x[18:]\n"
        "        # full_x = np.concatenate([x_c, self.best_x[18:]])\n"
        "        # self._evaluate(full_x, func)\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Hybrid Strategy: Use LHS to find a starting basin, then run CMA-ES on the 18 continuous dimensions while keeping the 6 glass IDs fixed. Periodically mutate the glass IDs using random swaps.",
        "Iterative Refinement: Optimize continuous variables using L-BFGS-B (via scipy.minimize) for 50 evals, then perform a discrete coordinate search on the 6 glass IDs. Repeat until budget is exhausted.",
        "Population-Based Search: Implement a simple Differential Evolution (DE) algorithm where the 18 continuous variables use DE mutation, but the 6 glass variables use uniform crossover to maintain discrete properties.",
        "Gradient-Biased Exploration: Use the provided `grad_func` to take a small initial step from the best LHS sample. Then, use that improved point as the centroid for a local CMA-ES search.",
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v3_Robust",
        n_parents=2,
        n_offspring=4,
        elitism=True,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [
        (s,) for s in range(1, 3)
    ]  # Reduced seeds for faster local evaluation
    test_seeds = [(s,) for s in range(11, 13)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=500,  # Reduced budget for 30b stability
        eval_timeout=600,
        name="DoubleGauss_v3",
        task_prompt=task_prompt,
        example_prompt=example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3")

    return Experiment(
        methods=[llamea],
        problems=[lens_problem],
        runs=1,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)

    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()
