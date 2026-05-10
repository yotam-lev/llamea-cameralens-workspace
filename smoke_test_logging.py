import os
import sys
import numpy as np
from typing import Optional

# Setup path
_FRAMEWORK_ROOT = "/Users/Yotam/Documents/School/University/25-26/Thesis/Thesis_Code/lens-blade-workspace/blade-framework"
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.experiment import Experiment
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation

def get_llm_mock():
    """Mock LLM to avoid actual API calls during smoke test."""
    class MockLLM:
        def __init__(self, *args, **kwargs):
            self.model = "mock-model"
        def set_logger(self, logger):
            pass
        def generate(self, prompt, **kwargs):
            return """
class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
    def __call__(self, func, grad_func=None):
        return 0.0, np.zeros(self.dim)
"""
        def sample_solution(self, session_messages, HPO=False):
            from llamea.solution import Solution
            return Solution(code="print('Hello World')", name="MockOptimizer")
        def to_dict(self):
            return {"model": self.model}
    return MockLLM()

def run_smoke_test():
    # Very small budget to finish quickly
    budget = 1 
    
    # Minimal prompts to verify they appear
    task_prompt = "SMOKE_TEST_TASK"
    initial_prompt = ["SMOKE_TEST_INIT"]
    example_prompt = "SMOKE_TEST_EXAMPLE"
    
    # Instantiate LLaMEA with kwargs to avoid direct keyword conflicts
    llamea = LLaMEA(
        get_llm_mock(),
        budget=budget,
        name="SMOKE_TEST_RUN",
        n_parents=1,
        n_offspring=1,
        elitism=False
    )

    lens_problem = ContextualLensOptimisation(
        training_instances=[(1,)],
        test_instances=[(1,)],
        budget_factor=10, 
        eval_timeout=10,  
        name="SmokeTest",
        example_prompt=example_prompt,
        task_prompt=task_prompt,
        initial_prompt=initial_prompt,
    )

    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/smoke_test")

    experiment = Experiment(
        methods=[llamea],
        problems=[lens_problem],
        runs=1,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=1,
    )

    print("Starting smoke test...")
    experiment()
    print("Smoke test complete. Check results/smoke_test for log.jsonl")

if __name__ == "__main__":
    run_smoke_test()
