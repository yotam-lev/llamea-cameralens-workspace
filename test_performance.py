
import time
import numpy as np
import sys
from pathlib import Path

# Add blade-framework and camera-lens-simulation to path
_PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT / "blade-framework"))
sys.path.insert(0, str(_PROJECT_ROOT / "camera-lens-simulation"))

from examples.double_gauss_objective import DoubleGaussObjective

def test_performance():
    print("Initializing DoubleGaussObjective...")
    start_init = time.time()
    obj = DoubleGaussObjective(enable_grad=True, enable_hessian=False)
    end_init = time.time()
    print(f"Initialization took {end_init - start_init:.2f} seconds.")

    x0_cont, x0_ids = obj.init_from_templates()
    theta = obj.pack_theta(x0_cont, x0_ids)
    
    print("\nTiming first objective call (includes JIT)...")
    start = time.time()
    f = obj.objective_theta(theta)
    end = time.time()
    print(f"First objective call took {end - start:.2f} seconds. Result: {f}")

    print("\nTiming 100 subsequent objective calls...")
    start = time.time()
    for _ in range(100):
        obj.objective_theta(theta)
    end = time.time()
    print(f"100 objective calls took {end - start:.4f} seconds (average {(end - start)/100:.6f}s).")

    print("\nTiming first gradient call (includes JIT)...")
    start = time.time()
    g = obj.gradient_cont_int(x0_cont, x0_ids)
    end = time.time()
    print(f"First gradient call took {end - start:.2f} seconds. Gradient shape: {g.shape}")

    print("\nTiming 100 subsequent gradient calls...")
    start = time.time()
    for _ in range(100):
        obj.gradient_cont_int(x0_cont, x0_ids)
    end = time.time()
    print(f"100 gradient calls took {end - start:.4f} seconds (average {(end - start)/100:.6f}s).")

if __name__ == "__main__":
    test_performance()
