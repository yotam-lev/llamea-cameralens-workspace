
import os
import sys
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.resolve()
CAMERA_LENS_ROOT = PROJECT_ROOT / "camera-lens-simulation"

if str(CAMERA_LENS_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMERA_LENS_ROOT))

# Ensure PYTHONPATH is set for subprocesses if needed
os.environ["PYTHONPATH"] = str(CAMERA_LENS_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from examples.double_gauss_objective import DoubleGaussObjective

def analyze():
    print("--- Analyzing grad0_cont (Baseline Design) ---")
    
    # 1. Initialize Objective
    obj = DoubleGaussObjective(enable_grad=True, enable_hessian=True)
    
    # 2. Get Initial Parameters
    x0_cont, x0_ids = obj.init_from_templates()
    
    # 3. Compute Gradient
    print(f"Continuous Vars (x0_cont) Shape: {x0_cont.shape}")
    print(f"Categorical Vars (x0_ids) Shape: {x0_ids.shape}")
    
    grad0_cont = obj.gradient_cont_int(x0_cont, x0_ids)
    
    print(f"\nGradient Shape: {grad0_cont.shape}")
    print(f"Gradient Norm: {np.linalg.norm(grad0_cont):.4f}")
    
    # Split the gradient to see Curvatures vs Thicknesses
    # Based on DoubleGaussObjective: 10 curvatures, 8 distances
    grad_curv = grad0_cont[:10]
    grad_dist = grad0_cont[10:]
    
    print("\n[GRADIENT COMPONENTS]")
    print(f"Curvature Gradients (First 5): {grad_curv[:5]}")
    print(f"Distance Gradients (First 5):  {grad_dist[:5]}")
    
    print("\n[MAGNITUDE COMPARISON]")
    print(f"Mean Abs Curvature Grad: {np.mean(np.abs(grad_curv)):.6f}")
    print(f"Mean Abs Distance Grad:  {np.mean(np.abs(grad_dist)):.6f}")

    # 4. Test Calling Patterns
    print("\n--- Testing Calling Patterns ---")
    
    # Pattern A: Direct call
    test_grad = obj.gradient_cont_int(x0_cont, x0_ids)
    print(f"Pattern A (Direct) Success: {np.allclose(test_grad, grad0_cont)}")
    
    # Pattern B: Perturbed call
    x_perturbed = x0_cont + np.random.normal(0, 0.01, x0_cont.shape)
    grad_perturbed = obj.gradient_cont_int(x_perturbed, x0_ids)
    print(f"Pattern B (Perturbed) Norm: {np.linalg.norm(grad_perturbed):.4f}")
    
    # Pattern C: Check if it's jittable/wrapped correctly
    import jax
    print(f"Is _f_grad a JAX function? {str(type(obj._f_grad))}")

if __name__ == "__main__":
    analyze()
