"""
Compare: no-context LLaMEA vs domain-aware LLaMEA.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from local_lens_problem import LocalLensOptimisation
from contextual_lens_problem import ContextualLensOptimisation
from config import get_llm, get_n_jobs

if __name__ == "__main__":
    llm = get_llm()

    budget = 50

    mutation_prompts = [
        "Refine the strategy of the selected algorithm to improve it.",
        "Generate a new algorithm that is different from the "
        "algorithms you have tried before.",
        "Refine and simplify the selected algorithm to improve it.",
    ]

    RS = RandomSearch(llm, budget=budget, name="RS_baseline")

    llamea_blind = LLaMEA(
        llm, budget=budget, name="LLaMEA_blind",
        n_parents=4, n_offspring=12, elitism=False,
        mutation_prompts=mutation_prompts,
    )

    llamea_informed = LLaMEA(
        llm, budget=budget, name="LLaMEA_informed",
        n_parents=4, n_offspring=12, elitism=False,
        mutation_prompts=mutation_prompts,
    )

    blind_problem = LocalLensOptimisation(
        budget_factor=2000, eval_timeout=60, name="Blind",
    )
    informed_problem = ContextualLensOptimisation(
        budget_factor=2000, eval_timeout=60, name="Informed",
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_comparison")

    experiment = Experiment(
        methods=[RS, llamea_blind, llamea_informed],
        problems=[blind_problem, informed_problem],
        runs=3,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=1,
    )

    experiment()