"""
V4: Gradient-Aware Memetic Evolution (Overnight Run)
Explicitly uses JAX gradients and strict API guardrails.
"""

import os
import sys
import inspect
from typing import Optional, Callable

# Ensure the blade-framework root is on sys.path
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA as BaseLLaMEAa
from iohblade.loggers import ExperimentLogger
from iohblade.problem import Problem
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs
from llamea import LLaMEA as LLAMEA_Algorithm

# Metadata for the run selector
RUN_META = {
    "name": "Lens V4 (Gradient-Aware Overnight)",
    "description": "Memetic LLaMEA with strict anti-crash guardrails for overnight run.",
    "context": True,
    "version": "v4_overnight",
}

class PhaseLLAMEA(LLAMEA_Algorithm):
    """
    Custom LLaMEA algorithm that uses specific prompts for seeding Gen 0,
    then switches to general mutation prompts for Gen 1+.
    """
    def __init__(self, *args, **kwargs):
        self.general_prompts = kwargs.pop('general_prompts', [])
        super().__init__(*args, **kwargs)
        self.initial_prompts = self.mutation_prompts.copy()

    def initialize_single(self):
        if not hasattr(self, '_init_prompt_idx'):
            self._init_prompt_idx = 0
        
        # Pick one specific prompt to guide this initial individual (Gen 0)
        guide = self.initial_prompts[self._init_prompt_idx % len(self.initial_prompts)]
        self._init_prompt_idx += 1
        
        # Temporarily modify task_prompt to seed the initial algorithm with this strategy
        old_task = self.task_prompt
        self.task_prompt = old_task + f"\n\n### STRATEGIC GOAL FOR THIS INITIAL VERSION:\n{guide}\n\nImplement the above strategy in your initial design."
        res = super().initialize_single()
        self.task_prompt = old_task
        return res

    def initialize(self):
        super().initialize()
        # After Gen 0 is created, switch to general mutation prompts for all subsequent evolution
        if self.general_prompts:
            self.logevent("Initialization complete. Reverting to general mutation prompts.")
            self.mutation_prompts = self.general_prompts

class LLaMEA_Phase(BaseLLaMEA):
    """
    Wrapper for PhaseLLAMEA to integrate with the BLADE Experiment framework.
    """
    def __call__(self, problem: Problem):
        if problem.logger_dir:
            import json
            import os
            prompts = {
                "role_prompt": "You are an elite algorithm designer specializing in mixed-variable optimization.",
                "task_prompt": problem.task_prompt,
                "example_prompt": problem.example_prompt,
                "output_format_prompt": problem.format_prompt,
                "initial_mutation_prompts": self.kwargs.get("mutation_prompts", []),
                "general_mutation_prompts": self.kwargs.get("general_prompts", []),
            }
            os.makedirs(problem.logger_dir, exist_ok=True)
            with open(os.path.join(problem.logger_dir, "prompts.json"), "w") as f:
                json.dump(prompts, f, indent=4)

        # Filter kwargs to only pass those supported by LLAMEA_Algorithm
        sig = inspect.signature(LLAMEA_Algorithm)
        valid_kwargs = {k: v for k, v in self.kwargs.items() if k in sig.parameters}
        
        self.llamea_instance = PhaseLLAMEA(
            f=problem,
            llm=self.llm,
            role_prompt="",
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,
            budget=self.budget,
            max_workers=1,
            general_prompts=self.kwargs.get('general_prompts', []),
            **valid_kwargs
        )
        return self.llamea_instance.run()

def configure_run(llm, n_jobs):
    budget = 50  # Evolutionary generations

    task_prompt = (
        "You are an elite algorithm designer specializing in mixed-variable optimization.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24-dimensional lens loss function. Parameters are strictly bounded in `[-1, 1]`.\n"
        "- `x[0:18]`: Continuous geometry.\n"
        "- `x[18:24]`: Categorical material IDs.\n\n"
        "### CRITICAL API USAGE (DO NOT DEVIATE):\n"
        "1. ALLOWED IMPORTS: You may ONLY import `numpy`, `scipy`, and `cma`. DO NOT import `sklearn` or any other external libraries. It will crash the environment.\n"
        "2. LHS SAMPLING: `lhs` is globally injected. DO NOT import it. Use `samples = lhs(n_samples=N, n_dim=self.dim)`. To scale to [-1, 1], manually compute `samples = samples * 2.0 - 1.0`. DO NOT use `qmc.scale`.\n"
        "3. SIGNATURES: `func(x)` and `grad_func(x)` accept EXACTLY ONE argument: a 24-dimensional numpy array. They return a scalar and a 24D gradient array, respectively.\n"
        "4. PARTIAL OPTIMIZATION & OS THREAD CRASH WARNING: When optimizing the 18 continuous variables with `scipy.optimize.minimize`, you MUST pass your custom gradient wrapper (`jac=grad_wrap`). If you omit the `jac` parameter, SciPy will fallback to numdiff, spawn thousands of threads, and CRASH THE SERVER.\n"
        "```python\n"
        "def cost_wrap(x_cont):\n"
        "    return func(np.concatenate([x_cont, x_disc]))\n\n"
        "def grad_wrap(x_cont):\n"
        "    if self.evals >= self.budget: return np.zeros(18)\n"
        "    return grad_func(np.concatenate([x_cont, x_disc]))[:18]\n\n"
        "res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18, options={'maxiter': 50})\n"
        "```\n"
        "5. CMA-ES PERSISTENCE: If you use CMA-ES, initialize `es` ONLY ONCE outside your main loop. `x0` must be a 1D numpy array, not an integer.\n"
        "6. NUMPY TYPING: If you build a new population in a Python list (`new_pop = []`), you MUST convert it back to a numpy array (`pop = np.array(new_pop)`) at the end of the generation.\n"
        "7. BUDGET: You MUST check `if self.evals >= self.budget: break` before EVERY call to `func(x)` or `grad_func(x)`.\n"
        "8. NO PARTIAL CODE: You MUST output the ENTIRE `class Optimizer:` definition, including `__init__`, `_evaluate`, and `__call__`.\n"
        "9. NO HARDCODED LOOPS: NEVER use `for generation in range(X):` for your main loop. Your main loop MUST be `while self.evals < self.budget:`.\n"
        "10. DISCRETE BOUNDS: When mutating glass IDs in phase 2, NEVER use integers like `randint(0, 10)`. The parameters are STRICTLY `[-1, 1]`. You must sample from `[-1.0, -0.5, 0.0, 0.5, 1.0]`.\n"
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
        "        # 1. Initialization (LHS scaled manually)\n"
        "        pop = lhs(n_samples=50, n_dim=self.dim)\n"
        "        pop = pop * 2.0 - 1.0\n"
        "        for x in pop: self._evaluate(x, func)\n"
        "\n"
        "        # Fallback if LHS didn't find a valid non-inf point\n"
        "        if self.best_f == float('inf'):\n"
        "            self.best_x = np.random.uniform(-1, 1, self.dim)\n"
        "\n"
        "        # 2. Main Loop\n"
        "        while self.evals < self.budget:\n"
        "            # Setup discrete glass vector using self.best_x as foundation\n"
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
        "            x0_18d = self.best_x[:18].copy() # Base geometry on best found\n"
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
        "Implement a trust-region approach: Use a global search to identify promising basins, then deploy a local search that respects the geometric bounds and uses gradients to 'descend' into the deep local optima characteristic of lens design.",
        "Scale your algorithm for a MASSIVE budget. Use a DE population size of at least 150 to 200 individuals to ensure massive global exploration before descending with L-BFGS-B."
    ]

    general_mutation_prompts = [
        "Refine the strategy of the selected solution to improve its performance and robustness.",
        "Propose structural changes to the algorithm to better explore the search space.",
        "Optimize the internal logic and parameters of the algorithm for faster convergence.",
        "Identify potential weaknesses in the current optimization approach and address them.",
        "Refine the current algorithm by introducing more sophisticated local search or mutation operators."
    ]

    llamea = LLaMEA_Phase(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic_Overnight",
        n_parents=4,
        n_offspring=12,
        elitism=True,  # CRITICAL: Ensures the best algorithms survive
        mutation_prompts=mutation_prompts,
        general_prompts=general_mutation_prompts
    )


    training_seeds = [(s,) for s in range(1, 3)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=500, 
        eval_timeout=600,  
        name="DoubleGauss_v4",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4_overnight")

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