"""
V3.2.1: The Island Model (Niching)
Maintains multiple parallel islands with fixed categorical variables. 
CMA-ES optimizes the continuous variables for each island independently.
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
    "name": "Lens V3.2.1",
    "description": "Island Model (Parallel CMA-ES with fixed niches)",
    "context": True,
    "version": "v3.2.1",
}

def configure_run(llm, n_jobs):
    budget = 20  # LLM Generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs.\n\n"
        "### CORE STRATEGY: THE ISLAND MODEL (NICHING)\n"
        "To avoid local optima caused by glass material changes, implement a parallel Island Model:\n"
        "1. Create `K=4` separate 'islands'. For each island, sample a random 18D continuous vector and a RANDOM DISCRETE 6D vector (using values like -1.0, -0.5, 0.0, 0.5, 1.0).\n"
        "2. Initialize `K` separate CMA-ES instances, one for each island.\n"
        "3. In your main optimization loop, iterate through the islands round-robin. For each island, ask its specific CMA-ES for a population of geometries, evaluate them combined with that island's fixed discrete glasses, and tell the fitnesses back to that specific CMA-ES.\n\n"
        "### STRICT API USAGE (DO NOT DEVIATE):\n"
        "1. LHS SAMPLING: `lhs` is globally injected. DO NOT import it (NEVER write `from pyDOE import lhs`). Use: `samples = lhs(n_samples=N, n_dim=self.dim)`.\n"
        "2. STATE SYNCING: `self._evaluate` automatically tracks `self.best_x` and `self.best_f`. NEVER manually overwrite `self.best_x` (e.g., `self.best_x = es.result[0]`), or you will delete the global best lens.\n"
        "3. CMA-ES PERSISTENCE: You MUST initialize your `K` CMA-ES instances ONLY ONCE, completely outside your main `while self.evals < self.budget:` loop. Inside the loop, only use `ask()` and `tell()`.\n"
        "4. WRAPPER & EVALUATION: When evaluating a CMA-ES continuous population `X`, combine it with the island's fixed discrete vector:\n"
        "   `fits = [self._evaluate(np.concatenate([x_c, island_discrete]), func) for x_c in X]`\n"
        "   `es.tell(X, fits)`\n"
        "5. BUDGET: You MUST check `if self.evals >= self.budget: break` before every evaluation.\n"
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
        "        # 1. Setup K islands and K CMA-ES instances here...\n"
        "        # 2. Main loop:\n"
        "        while self.evals < self.budget:\n"
        "             # Iterate through islands, ask(), evaluate(), tell()\n"
        "             pass\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Improve the Island Model: If a specific island's CMA-ES stops improving (stagnates for 5 iterations), kill that island and re-initialize it with a completely new random discrete glass vector to explore a new basin.",
        "Resource Allocation: Instead of round-robin, dynamically allocate more of the evaluation budget to the islands that have the lowest current fitness scores.",
    ]

    llamea = LLaMEA(
        llm, budget=budget, name="LLaMEA_v3_2_1", n_parents=2, n_offspring=4,
        elitism=True, mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 4)]
    test_seeds = [(s,) for s in range(11, 14)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds, test_instances=test_seeds,
        budget_factor=1000, eval_timeout=900, name="DoubleGauss_Island",
        task_prompt=task_prompt, example_prompt=example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3_2_1")

    return Experiment(
        methods=[llamea], problems=[lens_problem], runs=1,
        show_stdout=True, exp_logger=logger, budget=budget, n_jobs=n_jobs,
    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()