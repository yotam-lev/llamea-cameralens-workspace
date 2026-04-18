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
    "context": False,
    "version": "v4_1000_False",
}


def configure_run(llm, n_jobs):
    budget = 40  # Evolutionary generations

    task_prompt = (
    "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
    "### Problem Structure:\n"
    "Minimize a 24-dimensional lens loss function. Parameters are bounded in `[-1, 1]`.\n"
    "- `x[0:18]`: Continuous geometry.\n"
    "- `x[18:24]`: Categorical material IDs.\n\n"
    "### CRITICAL API USAGE (DO NOT DEVIATE):\n"
    "1. SIGNATURES: `func(x)` and `grad_func(x)` accept EXACTLY ONE argument: a 24-dimensional numpy array. They return a scalar and a 24D gradient array, respectively.\n"
    "2. PARTIAL OPTIMIZATION: When optimizing only the 18 continuous variables with `scipy.optimize.minimize`, NEVER pass `args=(x_disc,)`. You MUST wrap the functions to handle concatenation and gradient slicing:\n"
    "```python\n"
    "def cost_wrap(x_cont):\n"
    "    return func(np.concatenate([x_cont, x_disc]))\n\n"
    "def grad_wrap(x_cont):\n"
    "    # Slice the gradient to return only the 18 continuous components\n"
    "    return grad_func(np.concatenate([x_cont, x_disc]))[:18]\n\n"
    "res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18)\n"
    "```\n"
    "3. CMA-ES PERSISTENCE: If you use CMA-ES, initialize `es` ONLY ONCE outside your main loop. `x0` must be a 1D numpy array, not an integer.\n"
    "4. NUMPY TYPING: If you build a new population in a Python list (`new_pop = []`), you MUST convert it back to a numpy array (`pop = np.array(new_pop)`) at the end of the generation to prevent indexing crashes.\n"
    "5. BUDGET: You MUST check `if self.evals >= self.budget: break` before EVERY call to `func(x)` or `grad_func(x)`.\n"
    "6. NO PARTIAL CODE: You MUST output the ENTIRE `class Optimizer:` definition, including `__init__`, `_evaluate`, and `__call__`.\n"
    "7. NO HARDCODED LOOPS: NEVER use for generation in range(X): for your main loop. Your main loop MUST be while self.evals < self.budget: so you consume the massive budget we are giving you.\n"
    "8. DISCRETE BOUNDS: When mutating glass IDs in phase 2, NEVER use integers like randint(0, 10). The parameters are STRICTLY [-1, 1]. You must sample from [-1.0, -0.5, 0.0, 0.5, 1.0]."
)
    example_prompt = (
        "Write a completely self-contained Python class named exactly `Optimizer`.\n"
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
        "        # FORCE ROUNDING OF CATEGORICALS TO NEAREST 0.5 in [-1, 1]\n"
        "        eval_x = x.copy()\n"
        "        eval_x[18:24] = np.round(eval_x[18:24] * 2) / 2 \n"
        "        eval_x[18:24] = np.clip(eval_x[18:24], -1.0, 1.0)\n"
        "        f = func(eval_x)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = eval_x.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        # 1. Initialization (LHS)\n"
        "        pop = lhs(n_samples=10, n_dim=self.dim)\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "\n"
        "        # 2. Gradient Extraction (Slice to 18D!)\n"
        "        if grad_func is not None:\n"
        "            grad24 = grad_func(self.best_x)\n"
        "            grad18 = grad24[:18] # Only take continuous gradients\n"
        "            # ... Use grad18 to bias your search ...\n"
        "\n"
        "        # 3. Main Loop\n"
        "        while self.evals < self.budget:\n"
        "            # Implement Memetic / DE / CMA-ES / L-BFGS-B logic here...\n"
        "            pass\n"
        "\n"
        "        return self.best_f, self.best_x\n"
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
        "gradients to 'descend' into the deep local optima characteristic of lens design.",
        "Scale your algorithm for a MASSIVE budget. Use a DE population size of at least 100 to 200 individuals to ensure massive global exploration before descending with L-BFGS-B."
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic",
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
        budget_factor=10000,  # MATCH PRODUCTION BUDGET
        eval_timeout=900,  # Increased to handle 10k evals + local search
        name="DoubleGauss_v4",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4_False_10000")

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
