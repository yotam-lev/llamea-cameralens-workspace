"""
Lens Local Test: Verifying local Ollama code generation and large-budget sandbox stability.
"""

import os
import sys

# Ensure the blade-framework root is on sys.path
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs

# Metadata for the run selector
RUN_META = {
    "name": "Lens Local Test",
    "description": "Local test verifying Ollama code generation and sandbox stability under 50k budget",
    "context": True,
    "version": "local_test",
}


def configure_run(llm, n_jobs):
    budget = 100  # Generations (LLaMEA loop budget)

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable, black-box optimization.\n\n"
        "### Problem Physics & Landscape:\n"
        "Your task is to MINIMIZE a 24-dimensional camera lens design loss function with strictly bounded `[-1, 1]` parameters.\n"
        "- Indices `x[0:18]`: 18 Continuous parameters (lens curvatures and distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs.\n"
        "The landscape is highly non-convex and filled with infeasible 'cliffs' where invalid lenses return extremely high loss (`inf`).\n\n"
        "### THE GRADIENT ADVANTAGE (`grad0_cont`) ###\n"
        "You are provided with `grad0_cont` (shape: `(18,)`), the exact analytical gradient of the 18 continuous parameters at the baseline design. "
        "Use this to bias your initial population or take an initial step. "
        "WARNING: Because `grad0_cont` is 18D and your solutions are 24D, you MUST slice your target array before applying the gradient (e.g., `x[:18] -= lr * self.grad0_cont`) to prevent NumPy broadcast crashes.\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. LHS SAMPLING: The helper function `lhs` is already available in the global scope. DO NOT import `lhs` or `latin_hypercube_sampling` from `pyDOE` or any other external library (it will raise a ModuleNotFoundError). Just call it directly as `samples = lhs(n_samples=20, n_dim=self.dim)` using keyword arguments.\n"
        "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`.\n"
        "3. DIMENSIONS: grad_func MUST be called with a full 24-dimensional vector: `grad_func(full_x)`. It will return an 18-dimensional vector.\n"
        "4. DIMENSIONS: func MUST be called with a full 24-dimensional vector. Never pass an 18-dimensional vector to func.\n"
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
        '        """Wrapper to safely track budget and update best solution."""\n'
        "        if self.evals >= self.budget:\n"
        "            return float('inf')\n"
        "        f = func(x)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = x.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        # 1. Initialization (LHS)\n"
        "        initial_population = lhs(n_samples=10, n_dim=self.dim)\n"
        "        for x in initial_population:\n"
        "            self._evaluate(x, func)\n"
        "        # 2. Simple Random Search with bounds \n"
        "        for _ in range(self.budget - 10):\n"
        "            x = np.random.uniform(-1, 1, self.dim)\n"
        "            self._evaluate(x, func)\n"
        "        return self.best_f, self.best_x\n"
        "```\n\n"
    )

    mutation_prompts = [
        "Implement a simple random search with Gaussian mutation around the best found solution.",
        "Implement a simple simulated annealing optimizer that cools a temperature parameter and accepts worse solutions with decaying probability.",
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_Local_Test",
        n_parents=1,
        n_offspring=2,
        elitism=False,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(1,)]
    test_seeds = [(11,)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=50000,
        eval_timeout=600,
        name="DoubleGauss_Local_Test",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/Lens_Local_Test_2")

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
    print(f"Starting local test experiment: {RUN_META['name']}")
    experiment()
