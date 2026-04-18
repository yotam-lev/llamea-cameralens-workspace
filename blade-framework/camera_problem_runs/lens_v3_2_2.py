"""
V3.2.2: Memetic Basin Hopping
Randomizes the categorical variables to jump to new basins, then aggressively
optimizes the continuous variables using L-BFGS-B to find the basin minimum.
"""

import os
import sys

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs

RUN_META = {
    "name": "Lens V3.2.2",
    "description": "Memetic Basin Hopping (Large discrete jumps + L-BFGS-B)",
    "context": True,
    "version": "v3.2.2",
}

def configure_run(llm, n_jobs):
    budget = 20  # LLM Generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs.\n\n"
        "### CORE STRATEGY: MEMETIC BASIN HOPPING\n"
        "We want to rapidly sample completely different glass combinations and find the geometric minimum for each. Implement this loop:\n"
        "1. Generate a completely new, random discrete 6D vector for the glass IDs using valid step increments (e.g., `np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=6)`). DO NOT use small floats.\n"
        "2. Define a local wrapper function `eval_cont(x_c)` that concatenates a continuous 18D vector with your new discrete 6D vector and calls `self._evaluate`.\n"
        "3. Run `scipy.optimize.minimize` using `L-BFGS-B` on `eval_cont` to optimize the 18 continuous variables for this specific glass combination. Limit `maxiter` so it doesn't drain the budget.\n"
        "4. Repeat this jump-and-descend process until `self.budget` is exhausted.\n\n"
        "### STRICT API USAGE (DO NOT DEVIATE):\n"
        "1. LHS SAMPLING: `lhs` is globally injected. DO NOT import it (NEVER write `from pyDOE import lhs`). Use: `samples = lhs(n_samples=N, n_dim=self.dim)`.\n"
        "2. SCIPY MINIMIZE: You MUST pass your custom wrapper function to minimize. Example:\n"
        "   `res = minimize(eval_cont, x0_18d, method='L-BFGS-B', bounds=[(-1, 1)]*18, options={'maxfun': 50})`\n"
        "3. STATE SYNCING: `self._evaluate` automatically tracks `self.best_x` and `self.best_f`. NEVER manually overwrite `self.best_x` or you will delete the global best lens.\n"
        "4. BUDGET: You MUST check `if self.evals >= self.budget: break` inside your loops and wrapper functions to prevent budget overruns.\n"
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
        "        # 1. Initial LHS sampling...\n"
        "        # 2. Main Memetic Loop:\n"
        "        while self.evals < self.budget:\n"
        "             # Generate random discrete glass vector\n"
        "             # Define eval_cont wrapper\n"
        "             # Run minimize(L-BFGS-B)\n"
        "             pass\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Refine the Basin Hopping: Instead of a completely random jump, apply a mutation to the global `self.best_x[18:24]` to generate the next discrete glass vector. This explores basins near the current best solution.",
        "Add a gradient kickstart: Before running L-BFGS-B, use the `grad_func` to take one large gradient descent step on the continuous variables to point scipy in the right direction.",
    ]

    llamea = LLaMEA(
        llm, budget=budget, name="LLaMEA_v3_2_2", n_parents=2, n_offspring=4,
        elitism=True, mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 4)]
    test_seeds = [(s,) for s in range(11, 14)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds, test_instances=test_seeds,
        budget_factor=1000, eval_timeout=900, name="DoubleGauss_Memetic",
        task_prompt=task_prompt, example_prompt=example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3_2_2")

    return Experiment(
        methods=[llamea], problems=[lens_problem], runs=1,
        show_stdout=True, exp_logger=logger, budget=budget, n_jobs=n_jobs,
    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()