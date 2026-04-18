"""
V3.1: Full 24D Differential Evolution approach.
Forces the LLM to treat the entire space as continuous during evolution, 
but dynamically rounds the categorical variables right before evaluation.
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
    "name": "Lens V3.1",
    "description": "LLaMEA targeting Full 24D Differential Evolution",
    "context": True,
    "version": "v3.1",
}

def configure_run(llm, n_jobs):
    budget = 20

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs.\n\n"
        "### CORE STRATEGY: FULL 24D DIFFERENTIAL EVOLUTION\n"
        "Implement a Differential Evolution (DE) algorithm across ALL 24 dimensions simultaneously. "
        "Treat the entire space as continuous during the mutation and crossover phases. "
        "However, you MUST dynamically round indices `x[18:24]` to the nearest valid value ONLY right before calling `self._evaluate(x, func)`. "
        "This allows the population to smoothly swarm toward good glass combinations.\n\n"
        "### STRICT API USAGE (DO NOT DEVIATE):\n"
        "1. LHS SAMPLING: The `lhs` function is already injected globally into your environment. DO NOT import it (e.g., NEVER write `from pyDOE import lhs`). Call it exactly like this: `samples = lhs(n_samples=N, n_dim=self.dim)`.\n"
        "2. SCIPY MINIMIZE: `res = minimize(func, x0, method='L-BFGS-B', bounds=[(-1, 1)]*len(x0))`.\n"
        "3. BUDGET: You MUST check `if self.evals >= self.budget` before every call to `func(x)`.\n"
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
        "        # Round the categorical variables just before evaluation!\n"
        "        eval_x = x.copy()\n"
        "        eval_x[18:24] = np.round(eval_x[18:24])\n"
        "        \n"
        "        f = func(eval_x)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = eval_x.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        pop = lhs(n_samples=20, n_dim=self.dim)\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "        \n"
        "        # Implement Full 24D Differential Evolution here...\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Refine the Differential Evolution strategy by experimenting with different mutation strategies (e.g., DE/best/1/bin instead of DE/rand/1/bin).",
        "Introduce adaptive F and CR parameters into the Differential Evolution algorithm to help it dynamically balance exploration and exploitation.",
        "Add a local refinement phase: After running Differential Evolution for 80% of the budget, run L-BFGS-B on the continuous variables of the best solution to polish the final geometry.",
    ]

    llamea = LLaMEA(
        llm, budget=budget, name="LLaMEA_v3_1", n_parents=2, n_offspring=4,
        elitism=True, mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 13)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds, test_instances=test_seeds,
        budget_factor=500, eval_timeout=600, name="DoubleGauss_v3_1",
        task_prompt=task_prompt, example_prompt=example_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v3_1")

    return Experiment(
        methods=[llamea], problems=[lens_problem], runs=1,
        show_stdout=True, exp_logger=logger, budget=budget, n_jobs=n_jobs,
    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']} - {RUN_META['description']}")
    experiment()