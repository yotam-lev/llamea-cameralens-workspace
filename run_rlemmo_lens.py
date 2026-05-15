import os
import sys
from pathlib import Path

# 1. 🛑 FORCE THE 10-CORE LIMIT HERE 🛑
# This must happen before any multiprocessing or config modules are loaded.
os.environ["SLURM_CPUS_PER_TASK"] = "10"

# Add blade-framework to sys.path
_PROJECT_ROOT = Path(__file__).parent.resolve()
_BLADE_DIR = _PROJECT_ROOT / "blade-framework"
if str(_BLADE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLADE_DIR))

# Add camera-lens-simulation to sys.path (needed by Problem)
_SIM_DIR = _PROJECT_ROOT / "camera-lens-simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

# 2. Import your dynamic config variables
from config import get_n_jobs, get_llm, get_platform

from iohblade.experiment import Experiment
from iohblade.methods import RLEMMO
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation
from iohblade.problems.lens_optimisation import LensOptimisation

def run_rlemmo_hpc():
    platform_name = get_platform()
    workers = get_n_jobs()
    
    print("\n" + "="*60)
    print(f"🚀 RUNNING RLEMMO HPC MODE ON {platform_name.upper()}")
    print(f"⚙️  Workers (Cores) Allocated: {workers}")
    print("="*60)
    
    # 3. Setup Method (Scale up the population)
    # 1,000 explorers gives a much better global search for complex spaces
    rlemmo = RLEMMO(
        llm=get_llm(),
        budget=1, 
        pop_size=1000, 
        name="RLEMMO_HPC_Scaled"
    )
    
    # 4. Setup Problem (Scale up the budget)
    training_seeds = [(1,)] 
    test_seeds = [(11,)]
    
    lens_problem = LensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=1000000, # Increased budget for the larger population
        eval_timeout=600,
        name="DoubleGauss_RLEMMO_HPC"
    )

    # 5. Setup Logger
    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/rlemmo_hpc")

    # 6. Run Experiment in Parallel
    # runs=10 means 10 totally independent evolutions running at the same time.
    # n_jobs=workers tells it to use your 10 allocated cores to process them.
    experiment = Experiment(
        methods=[rlemmo],
        problems=[lens_problem],
        runs=10, 
        show_stdout=True,
        exp_logger=logger,
        budget=1,
        n_jobs=workers
    )

    print("Engaging Parallel Search...")
    experiment()
    print("\nHPC run completed. Check results/rlemmo_hpc/ for logs.")

if __name__ == "__main__":
    run_rlemmo_hpc()