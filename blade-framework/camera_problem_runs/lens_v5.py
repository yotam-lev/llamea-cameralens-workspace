"""
V5: Hardened Gradient-Aware Memetic Evolution.
Fixes: 
1. Robust __call__ signature for dry-runs.
2. Library restrictions (No sklearn/qmc).
3. Explicit boundary & casting logic.
4. Mandatory state initialization.
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
    "name": "Lens V5 (Hardened)",
    "description": "Hardened Memetic LLaMEA with strict library and signature guardrails",
    "context": False,
    "version": "v5_10000_Hardened",
}

def configure_run(llm, n_jobs):
    budget = 40  # Evolutionary generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### CRITICAL ENVIRONMENT RESTRICTIONS:\n"
        "1. ALLOWED LIBRARIES: Use ONLY `numpy`, `scipy`, and `cma`. NEVER import `sklearn` or `scipy.stats.qmc`.\n"
        "2. LHS SYNTAX: If you need Latin Hypercube Sampling, use this snippet:\n"
        "   `pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))` (or a custom numpy implementation).\n"
        "3. SIGNATURES: Your `__call__` MUST handle optional arguments to pass framework dry-runs:\n"
        "   `def __call__(self, func, grad_func=None, hess_func=None, **kwargs):` \n\n"
        "### PROBLEM STRUCTURE:\n"
        "Minimize a 24-dimensional lens loss function. Bounds: `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs (Must be rounded to steps of 0.5).\n\n"
        "### IMPLEMENTATION RULES:\n"
        "- WRAPPER FUNCTIONS: When using `scipy.optimize.minimize`, use a wrapper to concatenate `x_cont` (18D) and `x_disc` (6D).\n"
        "- BOUNDARY ENFORCEMENT: Inside `_evaluate` and after any mutation, you MUST clip values: `x = np.clip(x, -1.0, 1.0)`.\n"
        "- STATE: You MUST initialize `self.evals = 0` and `self.best_f = float('inf')` in `__init__`.\n"
        "- BUDGET: Always check `if self.evals >= self.budget: break` before calling `func` or `grad_func`."
    )

    example_prompt = (
        "Write a completely self-contained Python class named exactly `Optimizer`.\n"
        "```python\n"
        "import numpy as np\n"
        "from scipy.optimize import minimize\n\n"
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
        "        # Strict Boundary and Casting Enforcement\n"
        "        eval_x = np.clip(x.copy(), -1.0, 1.0)\n"
        "        eval_x[18:24] = np.round(eval_x[18:24] * 2.0) / 2.0\n"
        "        \n"
        "        f = func(eval_x)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = eval_x.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):\n"
        "        # Initialize population using standard numpy\n"
        "        pop = np.random.uniform(-1, 1, (10, self.dim))\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "\n"
        "        while self.evals < self.budget:\n"
        "            # Optimization logic here...\n"
        "            pass\n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Implement a Memetic Hybrid: Use Differential Evolution for global exploration of the 24D space. "
        "Every 5 generations, take the best individual and perform a local search on the 18 continuous variables "
        "using L-BFGS-B and the grad_func.",
        
        "Gradient-Guided Exploration: In your mutation step, move the 18 continuous dimensions in the direction "
        "of `-grad_func(x)` with a small step size, while using random discrete swaps for the remaining 6 dimensions.",
        
        "Categorical Refinement: Use a specialized crossover for dimensions 18-24 that only selects values from "
        "{-1.0, -0.5, 0.0, 0.5, 1.0} to ensure the physics engine receives valid glass IDs."
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v5_Hardened",
        n_parents=3,
        n_offspring=9,
        elitism=True,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=1000,
        eval_timeout=900,
        name="DoubleGauss_v5",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v5_Hardened_10000")

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
    print(f"Starting experiment: {RUN_META['name']}")
    experiment()