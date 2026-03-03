"""
Example: Use BLADE + LLaMEA to auto-design optimizers
for the camera lens design problem.
"""
from iohblade.experiment import Experiment
from iohblade.llm import OpenAI_LLM  # or Ollama_LLM, Claude_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.problems import Lensoptimisation
from iohblade.loggers import ExperimentLogger

# 1. Configure the LLM
llm = OpenAI_LLM("gpt-4o")  # Or: Ollama_LLM("qwen2.5-coder:14b")

# 2. Configure search methods
budget = 100  # Number of optimizer candidates to generate
RS = RandomSearch(llm, budget=budget)
llamea = LLaMEA(
    llm,
    budget=budget,
    name="LLaMEA_lens",
    n_parents=4,
    n_offspring=12,
    elitism=False,
)

# 3. Configure the lens problem
training_seeds = [(s,) for s in range(1, 6)]
test_seeds = [(s,) for s in range(6, 16)]

lens_problem = Lensoptimisation(
    training_instances=training_seeds,
    test_instances=test_seeds,
    budget_factor=5000,   # Each generated optimizer gets 5000 func evals
    eval_timeout=300,     # 5 min max per evaluation
    name="DoubleGauss",
)

# 4. Run experiment
logger = ExperimentLogger("results/lens_optimisation")
experiment = Experiment(
    methods=[RS, llamea],
    problems=[lens_problem],
    runs=5,
    show_stdout=True,
    exp_logger=logger,
)
experiment()