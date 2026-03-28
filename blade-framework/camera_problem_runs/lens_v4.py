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
    budget = 100 # Evolutionary generations

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
        "gradients to 'descend' into the deep local optima characteristic of lens design."
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v4_Memetic",
        n_parents=6,
        n_offspring=12,
        elitism=True,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 10)]
    test_seeds = [(s,) for s in range(11, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=5000, # MATCH PRODUCTION BUDGET
        eval_timeout=6000,   # Increased to handle 10k evals + local search
        name="DoubleGauss_v4",
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v4")

    return Experiment(
        methods=[llamea],
        problems=[lens_problem],
        runs=3,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=n_jobs,
    )

if __name__ == "__main__":
    experiment = configure_run(get_llm(), n_jobs=1)
    print(f"Starting experiment: {RUN_META['name']}")
    experiment()
