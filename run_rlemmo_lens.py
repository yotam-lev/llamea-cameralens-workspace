import os
import sys
from pathlib import Path

# Add blade-framework to sys.path
_PROJECT_ROOT = Path(__file__).parent.resolve()
_BLADE_DIR = _PROJECT_ROOT / "blade-framework"
if str(_BLADE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLADE_DIR))

# Add camera-lens-simulation to sys.path (needed by Problem)
_SIM_DIR = _PROJECT_ROOT / "camera-lens-simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from iohblade.experiment import Experiment
from iohblade.methods import RLEMMO, RandomSearch
from iohblade.loggers import ExperimentLogger
from contextual_lens_problem import ContextualLensOptimisation

def run_rlemmo_baseline():
    print("\n" + "="*60)
    print("RUNNING RLEMMO BASELINE ON DOUBLE GAUSS")
    print("="*60)
    
    # 1. Setup Method
    rlemmo = RLEMMO(budget=1, pop_size=50, name="RLEMMO_Baseline")
    
    # 2. Setup Problem
    training_seeds = [(1,)] 
    test_seeds = [(11,)]
    
    lens_problem = ContextualLensOptimisation(
        training_instances=training_seeds,
        test_instances=test_seeds,
        budget_factor=5000, 
        eval_timeout=600,
        name="DoubleGauss_RLEMMO_Test"
    )

    # 3. Setup Logger
    os.makedirs("results", exist_ok=True)
    logger = ExperimentLogger("results/rlemmo_baseline")

    # 4. Run Experiment
    experiment = Experiment(
        methods=[rlemmo],
        problems=[lens_problem],
        runs=1,
        show_stdout=True,
        exp_logger=logger,
        budget=1,
        n_jobs=1
    )

    print("Starting RLEMMO execution...")
    experiment()
    print("\nBaseline run completed. Check results/rlemmo_baseline/ for logs.")

if __name__ == "__main__":
    run_rlemmo_baseline()
