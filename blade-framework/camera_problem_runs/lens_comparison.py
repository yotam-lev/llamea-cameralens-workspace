"""
Compare: no-context LLaMEA vs domain-aware LLaMEA.
"""
import os
import sys

# Ensure the blade-framework root is on sys.path so that framework-level
# modules (local_lens_problem, contextual_lens_problem, …) are importable
# from this subdirectory.
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from local_lens_problem import LocalLensOptimisation
from contextual_lens_problem import ContextualLensOptimisation

# Metadata for the run selector
RUN_META = {
    "name": "Lens Comparison",
    "description": "Head-to-head: no-context LLaMEA vs domain-aware LLaMEA",
    "context": "both",
    "version": "comparison",
}


def configure_run(llm, n_jobs):
    """
    Return a fully configured Experiment ready to be called.

    Args:
        llm: Pre-initialized LLM instance from config.
        n_jobs: Worker count from config.

    Returns:
        iohblade.experiment.Experiment instance.
    """
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

    return Experiment(
        methods=[RS, llamea_blind, llamea_informed],
        problems=[blind_problem, informed_problem],
        runs=3,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=n_jobs,
    )
