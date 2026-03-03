"""
MVP: LLaMEA generates optimisers for the Double-Gauss lens problem.
"""
import os
import sys

# Ensure the blade-framework root is on sys.path so that framework-level
# modules (local_lens_problem, config, …) are importable from this subdirectory.
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from local_lens_problem import LocalLensOptimisation

# Metadata for the run selector
RUN_META = {
    "name": "Lens MVP",
    "description": "LLaMEA + RandomSearch on Double-Gauss (no domain context)",
    "context": False,
    "version": "mvp",
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
    budget = 20

    RS = RandomSearch(llm, budget=budget, name="RS_baseline")

    llamea = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA_mvp",
        n_parents=1,
        n_offspring=1,
        elitism=True,
        mutation_prompts=[
            "Refine the strategy of the selected algorithm to improve it.",
        ],
    )

    training_seeds = [(s,) for s in range(1, 6)]
    test_seeds = [(s,) for s in range(6, 16)]

    lens_problem = LocalLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=2000,
        eval_timeout=60,
        name="DoubleGauss_MVP",
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/lens_mvp")

    return Experiment(
        methods=[RS, llamea],
        problems=[lens_problem],
        runs=2,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=n_jobs,
    )
