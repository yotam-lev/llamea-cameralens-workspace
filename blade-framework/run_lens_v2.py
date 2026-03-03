"""
V2: Larger population, diverse mutations, tighter timeout,
    with domain-aware lens optimisation context.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs
if __name__ == "__main__":
    llm = get_llm()

    budget = 50

    RS = RandomSearch(llm, budget=budget, name="RS_baseline")

    mutation_prompts = [
        # Small refinement — tweak parameters, fix bugs
        "Refine the strategy of the selected algorithm to improve it.",
        # Exploration — force the LLM to try something structurally new
        "Generate a new algorithm that is different from the "
        "algorithms you have tried before.",
        # Simplification — sometimes less is more
        "Refine and simplify the selected algorithm to improve it.",
        # Domain-aware — nudge toward optics-specific strategies
        "The lens design landscape is highly multimodal with many "
        "infeasible regions. Refine the algorithm to better handle "
        "discrete glass material parameters and avoid local optima.",
    ]

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_v2",
        n_parents=4,
        n_offspring=12,
        elitism=False,
        mutation_prompts=mutation_prompts,
    )

    training_seeds = [(s,) for s in range(1, 6)]
    test_seeds = [(s,) for s in range(6, 16)]

    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=2000,
        eval_timeout=60,
        name="DoubleGauss_v2",
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_v2")

    experiment = Experiment(
        methods=[llamea],
        problems=[lens_problem],
        runs=3,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=1,
    )

    experiment()