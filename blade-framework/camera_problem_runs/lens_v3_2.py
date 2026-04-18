"""
V3.2: Alternating Co-Evolution approach.
Forces the LLM to alternate between optimizing continuous geometry (CMA-ES/L-BFGS-B) 
and discrete glass IDs (Hill Climbing/Random Search) in cycles.
"""

from math import lcm
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
    "name": "Lens V3.2",
    "description": "LLaMEA targeting Alternating Co-Evolution (CMA-ES + Discrete)",
    "context": True,
    "version": "v3.2",
}

def configure_run(llm, n_jobs):
    budget = 20

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs.\n\n"
        "### CORE STRATEGY: ALTERNATING CO-EVOLUTION\n"
        "Changing glass materials instantly ruins existing geometric alignments, causing infinite loss. "
        "To solve this, implement an Alternating Optimization loop:\n"
        "1. Phase 1 (Geometry): Freeze the 6 glass IDs. Run CMA-ES or L-BFGS-B on the 18 continuous variables for a small budget (e.g., 20 evals).\n"
        "2. Phase 2 (Materials): Freeze the 18 continuous variables. Run a discrete Neighborhood Search or Random Swap on the 6 glass IDs for a small budget.\n"
        "3. Repeat alternating phases until `self.budget` is exhausted.\n\n"
        "### STRICT API USAGE (DO NOT DEVIATE):\n"
        "1. LHS SAMPLING: The `lhs` function is already injected globally into your environment. DO NOT import it (e.g., NEVER write `from pyDOE import lhs`). Call it exactly like this: `samples = lhs(n_samples=N, n_dim=self.dim)`.\n"
        "2. CMA-ES: Use exactly this: `es = cma.CMAEvolutionStrategy(x0, sigma0, inopts={'popsize': N})`. Access results via `es.result[0]`.\n"
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
        "        pop = lhs(n_samples=10, n_dim=self.dim)\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "        \n"
        "        while self.evals < self.budget:\n"
        "             # Phase 1: Optimize Continuous (Keep best_x[18:] fixed)\n"
        "             # Phase 2: Optimize Discrete (Keep best_x[:18] fixed)\n"
        "             pass\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Refine the Alternating Strategy: Spend 70% of the budget on continuous geometry adaptation using CMA-ES, and 30% of the budget exploring new discrete glass ID combinations.",
        "Introduce a Local Search: During the discrete glass optimization phase, instead of purely random swaps, try perturbing only 1 or 2 glass IDs at a time (e.g., adding/subtracting a small integer).",
        "Use the provided `grad_func` to rapidly realign the continuous geometry (Phase 1) via a few gradient steps before switching back to testing new glass IDs (Phase 2).",
    ]

    llamea = LLaMEA(
        llm, budget=budget, name="LLaMEA_v3_2", n_parents=2, n_offspring=4,
        elitism=True, mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 13)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds, test_instances=test_seeds,
        budget_factor=500, eval_timeout=600, name="DoubleGauss_v3_2",
        task_prompt=task_prompt, example_prompt=example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3_2")

    return Experiment(
        methods=[llamea], problems=[lens_problem], runs=1,
        show_stdout=True, exp_logger=logger, budget=budget, n_jobs=n_jobs,
    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()