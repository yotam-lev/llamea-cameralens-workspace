"""
MVP: LLaMEA generates optimisers for the Double-Gauss lens problem.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from local_lens_problem import LocalLensOptimisation
from config import get_llm, get_n_jobs

if __name__ == "__main__":
    llm = get_llm()

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

    experiment = Experiment(
        methods=[RS, llamea],
        problems=[lens_problem],
        runs=2,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=1,
    )

    experiment()