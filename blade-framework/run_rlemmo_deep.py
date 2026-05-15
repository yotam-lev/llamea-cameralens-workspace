import os
import sys
from pathlib import Path

# 1. 🛑 STRICTLY ENFORCE CPU ALLOCATION LIMIT 🛑
os.environ["SLURM_CPUS_PER_TASK"] = "10"

_PROJECT_ROOT = Path(__file__).parent.resolve()
_BLADE_DIR = _PROJECT_ROOT / "blade-framework"
if str(_BLADE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLADE_DIR))

_SIM_DIR = _PROJECT_ROOT / "camera-lens-simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.problems.lens_optimisation import LensOptimisation
from iohblade.methods.rlemmo_deep import DeepRLEMMO

def problem_builder():
    """Factory function to bypass pickling issues in multiprocessing."""
    return LensOptimisation(
        training_instances=[(1,)],
        test_instances=[(11,)],
        budget_factor=1000000, 
        eval_timeout=600,
        name="DoubleGauss_DeepRL_Env"
    )

def run_deep_rl_hpc():
    print("\n" + "="*60)
    print("🚀 RUNNING DEEP RL RLEMMO HPC MODE")
    print("="*60)
    
    # 1. Setup the Method (DeepRL acts as our Agent and Environment Manager)
    # We allocate 10 environments to perfectly saturate the 10 CPU cores
    deep_rlemmo = DeepRLEMMO(
        problem_builder=problem_builder,
        num_envs=10, 
        pop_size=100, 
        name="DeepRLEMMO_PPO"
    )
    
    # 2. Setup the Dummy Problem for the Framework Logger
    # The actual problems are instantiated inside the Vectorized Environments
    dummy_problem = problem_builder()

    # 3. Setup Logger
    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/deep_rlemmo_hpc")

    # 4. Run Experiment
    experiment = Experiment(
        methods=[deep_rlemmo],
        problems=[dummy_problem],
        runs=1,  # PPO handles the massive parallel search internally
        show_stdout=True,
        exp_logger=logger,
        budget=1,
        n_jobs=1
    )

    print("Engaging GPU-Accelerated Search...")
    experiment()
    print("\nDeep RL run completed. Check results/deep_rlemmo_hpc/ for logs.")

if __name__ == "__main__":
    run_deep_rl_hpc()