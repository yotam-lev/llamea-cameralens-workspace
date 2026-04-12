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
    "version": "v4",
}


def configure_run(llm, n_jobs):
    budget = 20  # Evolutionary generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable, black-box optimization.\n\n"
        "### Problem Physics & Landscape:\n"
        "Your task is to MINIMIZE a 24-dimensional camera lens design loss function with strictly bounded `[-1, 1]` parameters.\n"
        "- Indices `x[0:18]`: 18 Continuous parameters (lens curvatures and distances).\n"
        "- Indices `x[18:24]`: 6 Categorical glass material IDs.\n"
        "The landscape is highly non-convex and filled with infeasible 'cliffs' where invalid lenses return extremely high loss (`inf`).\n\n"
        "### THE GRADIENT ADVANTAGE (`grad0_cont`) ###\n"
        "You are provided with `grad0_cont` (shape: `(18,)`), the exact analytical gradient of the 18 continuous parameters at the baseline design. "
        "Use this to bias your initial population or take an initial pseudo-gradient descent step. "
        "WARNING: Because `grad0_cont` is 18D and your solutions are 24D, you MUST slice your target array before applying the gradient (e.g., `x[:18] -= lr * self.grad0_cont`) to prevent NumPy broadcast crashes.\n\n"
        "### STRICT CODING STANDARDS (CRITICAL) ###\n"
        "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for best solution and `es.result[1]` for best fitness. To get population size, use `es.popsize`. Remember `es.ask()` returns a LIST of arrays.\n"
        "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`.\n"
        "3. LHS SAMPLING: When calling the Latin Hypercube sampler, you MUST use keyword arguments: `samples = lhs(n_samples=20, n_dim=self.dim)`.\n"
        "To initialize CMA-ES, you MUST use exactly the signature es = cma.CMAEvolutionStrategy(x0, sigma0, inopts={'popsize': N}). Do not use dim, Settings, or other hallucinated arguments. x0 must be a 1D numpy array, and sigma0 must be a float (e.g., 0.2)\n"
        "When using scipy.optimize.differential_evolution, you MUST use maxiter to limit evaluations. Do NOT use maxeval. The bounds must be a list of tuples, e.g., bounds=[(-1, 1)] * dim"
        "Do not reference variables like grad0_cont or initial_population before they are explicitly defined in the current method. If you need the gradient, extract it dynamically inside __call__ or pass it explicitly"
        "Give the LLM a rigid class structure it must fill in, rather than letting it design the __init__ and __call__ structures completely from scratch."
        "You MUST structure your __call__ method using the exact dimensionality management shown below."
        "1. grad_func MUST be called with a full 24-dimensional vector: grad_cont = grad_func(full_x). It will return an 18-dimensional vector.\n"
        "2. func MUST be called with a full 24-dimensional vector. Never pass an 18-dimensional vector to func.\n"
        "3. Do not manually write evaluation loops if using scipy.optimize. Map them through a wrapper function that tracks the budget.\n"
        "4. FEEDBACK: You can implement `receive_feedback(self, info)` to get structured data after each call, or use `print()` to send debugging info back to yourself.\n"
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
        "\n"
        "        # 2. Gradient Extraction (MUST PASS 24D VECTOR)\n"
        "        if grad_func is not None:\n"
        "            grad18 = grad_func(self.best_x) # Returns 18D\n"
        "            # ... Use grad18 to bias your search ...\n"
        "\n"
        "        # 3. Continuous Optimization (CMA-ES)\n"
        "        # ... Create 18D CMA-ES ...\n"
        "        # When evaluating CMA-ES samples, combine them with categorical variables:\n"
        "        # full_x = np.concatenate([x_cont_18, self.best_x[18:]])\n"
        "        # f = self._evaluate(full_x, func)\n"
        "\n"
        "        # 4. Categorical Optimization (DE / Local Search)\n"
        "        # ... Optmize indices [18:24] ...\n"
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
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic",
        n_parents=2,
        n_offspring=4,
        elitism=False,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=1000,  # MATCH PRODUCTION BUDGET
        eval_timeout=600,  # Increased to handle 10k evals + local search
        name="DoubleGauss_v4",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4")

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
