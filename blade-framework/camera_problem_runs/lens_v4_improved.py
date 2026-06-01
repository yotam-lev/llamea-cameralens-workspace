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
        "- Indices `x[0:18]`: 18 Continuous parameters (curvatures/distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs (must be integers within [0, N_materials-1]).\n"
        "The landscape contains 'cliffs' where invalid lenses return `inf`.\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. CATEGORICAL HANDLING: You MUST treat indices 18-24 as integers. Before passing ANY array to `func` or `grad_func`, "
        "you MUST apply a mapping: `x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)`. Failing to do this causes simulation crashes.\n"
        "2. GRADIENT ACCESS: When using gradients, slice them correctly. `grad_func(full_x)` returns an 18D gradient. Do not use this gradient to mutate categorical dimensions.\n"
        "3. CMA-ES ACCESS: Use `es.result[0]` for best x, `es.result[1]` for best fitness. `es.ask()` returns a LIST.\n"
        "4. SCIPY MINIMIZE: Use `scipy.optimize.minimize(..., jac=grad_func)`. Ensure `res.x` is clamped to valid categories before further use.\n"
        "5. WRAPPER: Use the provided `_evaluate` method to track evaluations. If `evals >= budget`, return `inf`.\n"
        "### LENS DESIGN TASK ###\n"
        "Memetic: Use CMA-ES on the continuous 18D space. For each offspring, round/clamp the 6 categorical indices before evaluation.",
        "Gradient-guided: Use `-grad0_cont` to perturb continuous variables. Resample categorical variables randomly from `[0, 5]` integers.",
        "Local Search: Identify promising basins via global search, then use SLSQP on the 18D subspace while keeping the best categorical configuration fixed.",
   
    
    )

    example_prompt = (
        "Write a completely self-contained Python class named `Optimizer`.\n"
        "```python\n"
        "import numpy as np\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n"
        "        self.evals = 0\n"
        "        self.best_f = float('inf')\n"
        "        self.best_x = np.zeros(dim)\n"
        "\n"
        "    def _map_categorical(self, x):\n"
        "        # Forces indices 18-24 to be valid integer categories [0, 5]\n"
        "        x_out = x.copy()\n"
        "        x_out[18:24] = np.clip(np.round(x_out[18:24]), 0, 5).astype(int)\n"
        "        return x_out\n"
        "\n"
        "    def _evaluate(self, x, func):\n"
        "        if self.evals >= self.budget: return float('inf')\n"
        "        x_clean = self._map_categorical(x)\n"
        "        f = func(x_clean)\n"
        "        self.evals += 1\n"
        "        if f < self.best_f:\n"
        "            self.best_f = f\n"
        "            self.best_x = x_clean.copy()\n"
        "        return f\n"
        "\n"
        "    def __call__(self, func, grad_func=None):\n"
        "        # Implementation here using self._evaluate(x, func)\n"
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
        n_parents=4,
        n_offspring=12,
        elitism= elitism_flag,
        mutation_prompts=mutation_prompts,
    )



    lens_problem = ContextualLensOptimisation(
        training_instances=[(s,) for s in range(1, 3)],
        test_instances=[(s,) for s in range(11, 16)],
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
