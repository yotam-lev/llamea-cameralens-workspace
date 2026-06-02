"""
V4 Improved: Gradient-Aware Memetic Evolution with Robust Categorical Handling.
Explicitly forces categorical variables to be mapped to valid integer indices.
"""

import os
import sys
from datetime import datetime

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
    "name": "Lens V4 Improved (Robust Categorical)",
    "description": "Memetic LLaMEA with gradient-aware 18D continuous search and clamped categorical mapping",
    "context": True,
    "version": "v4.1",
}

def configure_run(llm, n_jobs):
    budget = 150  # Evolutionary generations
    budget_factor = 50000
    version_no = "v4.1"
    elitism_flag = False

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable, black-box optimization.\n\n"
        "### Problem Physics & Landscape:\n"
        "MINIMIZE a 24-dimensional camera lens design loss function.\n"
        "The entire search space is normalized. EVERY dimension MUST be bounded strictly within [-1.0, 1.0].\n"
        "- Indices `x[0:18]`: 18 Continuous geometry parameters (curvatures/distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs.\n"
        "The landscape is highly non-convex and contains 'cliffs' where invalid lenses return `inf`.\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. NO MANUAL DISCRETIZATION: Treat all 24 dimensions as continuous floats in [-1.0, 1.0]. "
        "Do NOT attempt to round, bin, or discretize indices [18:24] yourself. The target environment automatically "
        "handles the mapping and integer projection of the glass catalogs internally. Just pass floats in [-1.0, 1.0].\n"
        "2. GRADIENT DIMENSIONALITY: `grad_func(full_24d_x)` requires a 24D array but returns an 18D gradient array. "
        "This gradient ONLY applies to the first 18 continuous variables.\n"
        "3. SCIPY MINIMIZE: If using `scipy.optimize.minimize` (like L-BFGS-B) on the continuous variables, "
        "you MUST create wrapper functions that accept an 18D array, concatenate it with the fixed 6D categorical array, "
        "and pass the full 24D array to the main `func` and `grad_func`.\n"
        "4. LHS SAMPLING: Always use keyword arguments: `lhs(n_samples=N, n_dim=24)`.\n"
        "5. WRAPPER: Always use your internal `self._evaluate(x, func)` to track the budget. If `evals >= budget`, return `inf`.\n"
    )

    example_prompt = (
        "Write a completely self-contained Python class named `Optimizer`.\n"
        "```python\n"
        "import numpy as np\n"
        "from scipy.optimize import minimize\n\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n"
        "        self.evals = 0\n"
        "        self.best_f = float('inf')\n"
        "        self.best_x = np.random.uniform(-1, 1, dim)\n"
        "\n"
        "    def _evaluate(self, x, func):\n"
        "        if self.evals >= self.budget: return float('inf')\n"
        "        # Ensure strictly bounded within [-1, 1]\n"
        "        x_clean = np.clip(x, -1.0, 1.0)\n"
        "        f = func(x_clean)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = x_clean.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        # Example: Local gradient refinement on the 18D continuous subspace\n"
        "        if grad_func is not None:\n"
        "            fixed_cats = self.best_x[18:24].copy()\n"
        "            \n"
        "            def cost_wrap(x_cont):\n"
        "                full_x = np.concatenate([x_cont, fixed_cats])\n"
        "                return self._evaluate(full_x, func)\n"
        "                \n"
        "            def grad_wrap(x_cont):\n"
        "                full_x = np.concatenate([x_cont, fixed_cats])\n"
        "                return grad_func(full_x) # Returns 18D gradient\n"
        "                \n"
        "            # Scipy strictly requires 1D float64 arrays\n"
        "            x0 = self.best_x[:18].astype(np.float64)\n"
        "            minimize(cost_wrap, x0, jac=grad_wrap, method='L-BFGS-B', bounds=[(-1.0, 1.0)] * 18)\n"
        "\n"
        "        # Add Global / Categorical search logic here...\n"
        "        \n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve its performance and robustness.",
        "Propose structural changes to the algorithm to better explore the search space.",
        "Optimize the internal logic and parameters of the algorithm for faster convergence.",
        "Identify potential weaknesses in the current optimization approach and address them.",
        "Refine the current algorithm by introducing more sophisticated local search or mutation operators."
        ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Improved",
        n_parents=3,
        n_offspring=8,
        elitism= elitism_flag,
        mutation_prompts=mutation_prompts,
    )



    lens_problem = ContextualLensOptimisation(
        training_instances=[(s,) for s in range(1,)],
        test_instances=[(s,) for s in range(11, 13)],
        budget_factor=budget_factor,
        eval_timeout=1800,
        name="DoubleGauss_v4.1",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"lens_v{version_no}_{budget}_{budget_factor}_{elitism_flag}{timestamp}"
    logger = ExperimentLogger(f"results/{dir_name}")
    

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
