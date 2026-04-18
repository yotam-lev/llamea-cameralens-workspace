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
    "context": True,
    "version": "v4_Long",
}

def configure_run(llm, n_jobs):
    budget = 50  # Evolutionary generations

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
        "    if self.evals >= self.budget: return np.zeros(18) # Safety check\n"
        "    # Slice the gradient to return only the 18 continuous components\n"
        "    return grad_func(np.concatenate([x_cont, x_disc]))[:18]\n\n"
        "res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18)\n"
        "```\n"
        "3. CRASH WARNING: When using `scipy.optimize.minimize`, you MUST pass your custom gradient wrapper (`jac=grad_wrap`). If you omit the `jac` parameter, the framework will run out of OS threads and crash the server.\n"
        "4. CMA-ES PERSISTENCE: If you use CMA-ES, initialize `es` ONLY ONCE outside your main loop. `x0` must be a 1D numpy array, not an integer.\n"
        "5. NUMPY TYPING: If you build a new population in a Python list (`new_pop = []`), you MUST convert it back to a numpy array (`pop = np.array(new_pop)`) at the end of the generation to prevent indexing crashes.\n"
        "6. BUDGET: You MUST check `if self.evals >= self.budget: break` before EVERY call to `func(x)` or `grad_func(x)`.\n"
        "7. NO PARTIAL CODE: You MUST output the ENTIRE `class Optimizer:` definition, including `__init__`, `_evaluate`, and `__call__`.\n"
        "8. NO HARDCODED LOOPS: NEVER use `for generation in range(X):` for your main loop. Your main loop MUST be `while self.evals < self.budget:` so you consume the massive budget we are giving you.\n"
        "9. DISCRETE BOUNDS: When mutating glass IDs in phase 2, NEVER use integers like `randint(0, 10)`. The parameters are STRICTLY `[-1, 1]`. You must sample from `[-1.0, -0.5, 0.0, 0.5, 1.0]`.\n"
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
        "        # 2. Main Loop\n"
        "        while self.evals < self.budget:\n"
        "            # Setup discrete glass vector\n"
        "            x_disc = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=6)\n"
        "\n"
        "            # MANDATORY WRAPPERS FOR L-BFGS-B\n"
        "            def cost_wrap(x_cont):\n"
        "                return self._evaluate(np.concatenate([x_cont, x_disc]), func)\n"
        "            \n"
        "            def grad_wrap(x_cont):\n"
        "                if self.evals >= self.budget: return np.zeros(18)\n"
        "                return grad_func(np.concatenate([x_cont, x_disc]))[:18]\n"
        "\n"
        "            x0_18d = np.random.uniform(-1, 1, size=18)\n"
        "            \n"
        "            # MUST PASS jac=grad_wrap TO PREVENT THREAD CRASH\n"
        "            if grad_func is not None:\n"
        "                res = minimize(cost_wrap, x0_18d, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18, options={'maxiter': 50})\n"
        "            \n"
        "            if self.evals >= self.budget: break\n"
        "\n"
        "        return self.best_f, self.best_x\n"
        "```\n\n"
    )

    mutation_prompts = [
        "Implement a memetic strategy: Use a population-based global explorer (like DE) to search the full 24D space. For the best individuals in each generation, apply a local gradient-based 'polish' (like L-BFGS-B) to only the first 18 dimensions using the provided grad_func.",
        "Design a mutation operator that uses the gradient. Instead of random noise, perturb the first 18 dimensions proportionally to the negative gradient direction (-grad_func(x)) to accelerate convergence toward the local minimum.",
        "The landscape is mixed-variable. Refine the algorithm to treat the last 6 dimensions as purely categorical (using discrete operators like crossover or rounding) while treating the first 18 as a smooth, differentiable manifold to be optimized via first-order information.",
        "Implement a trust-region approach: Use the global search to identify promising basins, then deploy a local search that respects the geometric bounds and uses gradients to 'descend' into the deep local optima characteristic of lens design.",
        "Scale your algorithm for a MASSIVE budget. Use a DE population size of at least 100 to 200 individuals to ensure massive global exploration before descending with L-BFGS-B."
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic_long",
        n_parents=4,
        n_offspring=12,
        elitism=True,  # SET TO TRUE SO THE BEST ALGORITHMS SURVIVE
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=50000, 
        eval_timeout=14400,  
        name="DoubleGauss_v4",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4_long")

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