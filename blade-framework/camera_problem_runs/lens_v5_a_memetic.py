"""
V5 A: CMA-ES + Trust-Region Hessian Memetic Search
"""
import os
import sys

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path: sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs

RUN_META = {"name": "Lens V5A (Memetic Hessian)", "description": "CMA-ES global + Trust-Constr local", "context": True, "version": "v5a"}

def configure_run(llm, n_jobs):
    task_prompt = (
        "You are an elite algorithm designer specializing in MINLP.\n\n"
        "### Problem Structure:\n"
        "Minimize a 24D function. `x[0:18]` are continuous, `x[18:24]` are categorical materials.\n\n"
        "### STRICT SKELETON TO AVOID CRASHES (DO NOT DEVIATE):\n"
        "```python\n"
        "import numpy as np\n"
        "from scipy import optimize\n"
        "import cma\n\n"
        "class Optimizer:\n"
        "    def __init__(self, budget: int, dim: int, **kwargs):\n"
        "        self.budget = budget\n"
        "        self.dim = dim\n"
        "        self.evals = 0\n"
        "        self.x0_baseline = np.zeros(dim)\n"
        "        if 'x0_baseline' in kwargs: self.x0_baseline = kwargs['x0_baseline']\n\n"
        "    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):\n"
        "        best_f, best_x = float('inf'), self.x0_baseline.copy()\n\n"
        "        # 1. CMA-ES requires an array for x0, NOT an int.\n"
        "        # es = cma.CMAEvolutionStrategy(list(self.x0_baseline), 0.2)\n"
        "        \n"
        "        # 2. DISCRETE RULES: Material IDs must be forced to valid states:\n"
        "        # x[18:] = np.clip(np.round(x[18:] * 2.0) / 2.0, -1.0, 1.0)\n\n"
        "        # 3. LOCAL SEARCH WRAPPERS (Only use if hess_func is not None):\n"
        "        # def cost_wrap(x_cont): return func(np.concatenate([x_cont, best_x[18:]]))\n"
        "        # def grad_wrap(x_cont): return grad_func(np.concatenate([x_cont, best_x[18:]]))[:18]\n"
        "        # def hess_wrap(x_cont): return hess_func(np.concatenate([x_cont, best_x[18:]]))[:18, :18]\n"
        "        # res = optimize.minimize(cost_wrap, best_x[:18], method='trust-constr', jac=grad_wrap, hess=hess_wrap, bounds=[(-1,1)]*18)\n\n"
        "        # ALWAYS check `if self.evals >= self.budget: break` before calling func.\n"
        "        return best_f, best_x\n"
        "```\n"
        "Your task: Implement a memetic algorithm. Run CMA-ES to explore materials, then polish the best found continuous parameters using the Hessian wrappers."
    )

    llamea = LLaMEA(llm, budget=30, name="LLaMEA_v5A", n_parents=2, n_offspring=4, elitism=False, task_prompt=task_prompt)
    lens_problem = ContextualLensOptimisation([(1,),(2,)], [(11,),(12,)], budget_factor=1000, eval_timeout=900, name="DoubleGauss_v5A")
    return Experiment([llamea], [lens_problem], 1, True, ExperimentLogger("results/lens_v5a"), 30, n_jobs)

if __name__ == "__main__":
    configure_run(get_llm(), 1)()