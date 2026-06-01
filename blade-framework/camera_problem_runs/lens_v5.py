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
from datetime import datetime 

# Ensure the blade-framework root is on sys.path
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)



from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from iohblade.problems.lens_optimisation import LensOptimisation
from config import get_llm, get_n_jobs
from contextual_lens_problem import ContextualLensOptimisation

# Metadata for the run selector
RUN_META = {
    "name": "Lens V5 (Hardened)",
    "description": "Hardened Memetic LLaMEA with strict library and signature guardrails",
    "context": False,
    "version": "v5_500_3",
}

def configure_run(llm, n_jobs):
    budget = 100
    budget_factor = 500
    version_no = "v5"
    elitism_flag = True
     # Evolutionary generations

    task_prompt = (
            "You are an elite algorithm designer specializing in mixed-variable, second-order optimization.\n\n"
            "### CRITICAL ENVIRONMENT & SYNTAX GUARDRAILS:\n"
            "1. ALLOWED LIBRARIES: Use ONLY `numpy`, `scipy`, and `cma`. NEVER import `sklearn` or `scipy.stats.qmc`.\n"
            "2. LHS SYNTAX: For Latin Hypercube Sampling, you must use standard numpy: `pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))`.\n"
            "3. SIGNATURES: Your `__call__` MUST exactly match this signature to pass framework dry-runs:\n"
            "   `def __call__(self, func, grad_func=None, hess_func=None, **kwargs):` \n"
            "4. STATE & BUDGET: Initialize `self.evals = 0` and `self.best_f = float('inf')` in `__init__`. Always check `if self.evals >= self.budget: break` before calling `func`, `grad_func`, or `hess_func`.\n"
            "5. BOUNDARY ENFORCEMENT (CRITICAL): Inside your evaluation wrapper, you MUST strictly clip values: `x = np.clip(x, -1.0, 1.0)`. The categorical variables `x[18:24]` MUST be rounded to steps of 0.5 (e.g., `x[18:24] = np.round(x[18:24] * 2.0) / 2.0`).\n\n"
            "### PROBLEM STRUCTURE:\n"
            "Minimize a 24-dimensional highly non-convex optical loss function. Bounds are strictly `[-1, 1]`.\n"
            "- `x[0:18]`: Continuous geometry.\n"
            "- `x[18:24]`: Categorical material IDs.\n\n"
            "### THE HESSIAN ADVANTAGE (`hess_func`) ###\n"
            "You are provided with `hess_func(full_24D_x)`. It returns an exact `(18, 18)` symmetric matrix of second derivatives for the continuous parameters.\n"
            "- **CRITICAL SOLVER RULE**: If using `scipy.optimize.minimize`, you MUST use `method='trust-constr'` because it is the ONLY exact-Hessian solver that supports `bounds`. Do NOT use `trust-exact` or `Newton-CG` with bounds.\n"
            "- **REGULARIZATION**: The landscape is non-convex. The exact Hessian will often be indefinite. You MUST regularize it (e.g., `H = hess_func(x) + 1e-3 * np.eye(18)`) to make it positive-definite before use.\n\n"
            "### INSPIRATION & BASELINE STRATEGIES (For Initial Guidance) ###\n"
            "To get started, consider these proven strategies for this specific landscape:\n"
            "- **Memetic Hybrids:** Use Differential Evolution (or CMA-ES) for global exploration of the 24D space. Periodically trigger local search (`scipy.optimize.minimize` with `trust-constr`) on the continuous subspace for the best individuals.\n"
            "- **Second-Order Mutations:** Instead of random noise, perturb the 18 continuous dimensions using a damped, regularized Newton step: `np.linalg.solve(H_reg, -grad)`.\n"
            "- **Categorical Refinement:** Use specialized crossover for dimensions 18-24 that strictly sample from valid ID steps ({-1.0, -0.5, 0.0, 0.5, 1.0}).\n"
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
        "        # Example of using the exact Hessian in a local search:\n"
        "        if hess_func is not None and grad_func is not None:\n"
        "            res = minimize(\n"
        "                lambda x_c: func(np.concatenate([x_c, self.best_x[18:]])),\n"
        "                self.best_x[:18],\n"
        "                jac=lambda x_c: grad_func(np.concatenate([x_c, self.best_x[18:]])),\n"
        "                hess=lambda x_c: hess_func(np.concatenate([x_c, self.best_x[18:]])),\n"
        "                method='trust-exact'\n"
        "            )\n"            
        "        while self.evals < self.budget:\n"
        "            # Optimization logic here...\n"
        "            pass\n"
        "        return self.best_f, self.best_x\n"
        "```\n"
    )

    mutation_prompts = [
        # Strategy 1: Escape & Exploitation
        "Critically analyze the algorithmic structure you just generated. Propose a novel exploration strategy that better escapes deep, infeasible local minima in the continuous subspace, while still utilizing the exact Hessian for rapid convergence when trapped in a promising basin.",
        
        # Strategy 2: Mixed-Variable Symbiosis
        "Design a more sophisticated mechanism for handling the mixed-variable nature of this problem. Instead of treating the categorical and continuous variables as entirely separate pipelines, how can the discrete choices (glass materials) dynamically dictate the continuous geometric optimization, or vice versa?",
        
        # Strategy 3: Adaptive Mechanisms
        "Introduce a self-tuning or adaptive mechanism. For example, dynamically adjust the frequency of local search, the mutation scale, or the population size based on runtime feedback like the improvement rate, budget remaining, or the condition number of the Hessian matrix.",
        
        # Strategy 4: Paradigm Shift
        "Take a completely different evolutionary or meta-heuristic paradigm (e.g., Swarm Intelligence, Simulated Annealing, or a Multi-Armed Bandit selector for operators) and adapt it specifically to exploit this 24D, second-order-enabled landscape."
    ]
    

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v5_x3",
        n_parents=3,
        n_offspring=9,
        elitism=elitism_flag,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=budget_factor,
        eval_timeout=1200,
        name="DoubleGauss_v5",
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