import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup paths to find the simulation and framework
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CAMERA_LENS_ROOT = os.path.join(PROJECT_ROOT, "camera-lens-simulation")
BLADE_FRAMEWORK_ROOT = os.path.join(PROJECT_ROOT, "blade-framework")

if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)
if BLADE_FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, BLADE_FRAMEWORK_ROOT)

from examples.double_gauss_objective import DoubleGaussObjective
from iohblade.problems.lens_optimisation import LHSWrapper

def load_optimizer(json_filename=None):
    """Loads a specific generated optimizer JSON file from extracted_gen_data or its subfolders."""
    extracted_dir = os.path.join(PROJECT_ROOT, "Lens_v4_50000_F/extracted_gen_data")
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Directory {extracted_dir} does not exist.")
    
    # Recursively find all .json files
    files = []
    for root, _, filenames in os.walk(extracted_dir):
        for f in filenames:
            if f.endswith(".json"):
                files.append(os.path.relpath(os.path.join(root, f), extracted_dir))
    
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {extracted_dir} or its subfolders.")
        
    if json_filename:
        # Match by filename or partial name
        matched = [f for f in files if json_filename in f]
        if matched:
            target_file = os.path.join(extracted_dir, matched[0])
        else:
            raise FileNotFoundError(f"Could not find file matching '{json_filename}' in {extracted_dir} or subfolders.")
    else:
        # Default to the first one
        target_file = os.path.join(extracted_dir, files[0])
        
    print(f"Loading optimizer code from: {os.path.basename(target_file)}")
    with open(target_file, "r") as f:
        data = json.load(f)
    return data["code"], data.get("id", "unknown")

def extract_optimizer_class(code):
    """Robustly extracts the optimizer class from the executed code string."""
    lhs_tool = LHSWrapper()
    exec_env = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "latin_hypercube_sampling": lhs_tool,
        "lhs": lhs_tool,
    }
    
    exec(code, exec_env)
    
    # Try finding exact class name 'Optimizer'
    OptimizerClass = exec_env.get("Optimizer")
    if not OptimizerClass:
        # Robust fallback extraction for custom class names
        ignore_list = ["LHSWrapper", "DoubleGaussObjective", "Solution", "Problem", "Optimizer"]
        for name, val in exec_env.items():
            if isinstance(val, type) and name not in ignore_list:
                if any(hasattr(val, m) for m in ["optimize", "solve", "run", "minimize", "__call__"]):
                    OptimizerClass = val
                    break
                    
    if not OptimizerClass:
        raise AttributeError("No valid Optimizer class could be extracted from the code.")
    return OptimizerClass

def main():
    parser = argparse.ArgumentParser(description="Aligned Camera Lens Optimization Runner & Visualizer")
    parser.add_argument("--optimizer", help="Filename or partial ID of the JSON file in extracted_gen_data/ (defaults to first file)")
    parser.add_argument("--budget", type=int, default=50000, help="Optimization budget (evaluations limit, default: 50000)")
    parser.add_argument("--seed", type=int, default=25, help="Random seed for optimization (default: 25)")
    args = parser.parse_args()
    
    print("=" * 70)
    print(" ALIGNED DOUBLE-GAUSS LENS OPTIMIZER & VISUALIZER ")
    print("=" * 70)
    
    # 2. Load the LLM-generated Optimizer code
    try:
        opt_code, opt_id = load_optimizer(args.optimizer)
        OptimizerClass = extract_optimizer_class(opt_code)
    except Exception as e:
        print(f"Error loading optimizer: {e}")
        sys.exit(1)
        
    # 3. Initialize double-gauss objective in JAX/Simulation root
    print("Initializing Double-Gauss Objective...")
    obj = DoubleGaussObjective(enable_grad=True, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta
    
    # Extract template initials for baseline gradient
    x0_cont, x0_ids = obj.init_from_templates()
    grad0_cont = obj.gradient_cont_int(x0_cont, x0_ids)
    
    # Setup baseline continuous variables gradient step scale
    scale = (ub - lb) / 2.0
    
    # 4. Instantiate the Optimizer using LLaMEA's signature bindings
    import inspect
    sig = inspect.signature(OptimizerClass.__init__)
    init_params = sig.parameters
    
    kwargs = {}
    if "budget" in init_params:
        kwargs["budget"] = args.budget
    if "dim" in init_params:
        kwargs["dim"] = dim
    if "grad0_cont" in init_params:
        kwargs["grad0_cont"] = grad0_cont
        
    if not kwargs:
        num_args = len(init_params) - 1
        if num_args >= 3:
            optimizer_inst = OptimizerClass(args.budget, dim, grad0_cont)
        elif num_args == 2:
            optimizer_inst = OptimizerClass(args.budget, dim)
        else:
            optimizer_inst = OptimizerClass(args.budget)
    else:
        optimizer_inst = OptimizerClass(**kwargs)
        
    # 5. Define Aligned Function and Gradient Wrappers
    def bounded_func(xn):
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        f_val = obj.objective_theta(np.clip(xr, lb, ub))
        if hasattr(optimizer_inst, "receive_feedback"):
            try:
                optimizer_inst.receive_feedback({"loss": f_val, "x_normalized": xn})
            except:
                pass
        return f_val

    def bounded_grad(xn):
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        xc, xi = obj.split_theta(np.clip(xr, lb, ub))
        g_val = obj.gradient_cont_int(xc, xi) * scale[:18]
        if hasattr(optimizer_inst, "receive_feedback"):
            try:
                optimizer_inst.receive_feedback({"grad": g_val, "x_normalized": xn})
            except:
                pass
        return g_val

    # Force-inject global properties inside the instance
    optimizer_inst.func = bounded_func
    optimizer_inst.grad_func = bounded_grad
    if not hasattr(optimizer_inst, "receive_feedback"):
        optimizer_inst.receive_feedback = lambda x: None
        
    np.random.seed(args.seed)
    
    # 6. Run the Optimization
    print(f"Running optimization (Budget: {args.budget}, Seed: {args.seed})...")
    start_time = time.time()
    
    # Select best execution entry method based on LLaMEA priority
    best_f = float('inf')
    best_x_normalized = np.zeros(dim)
    
    entry_methods = ["__call__", "optimize", "solve", "run", "minimize"]
    executed = False
    for method_name in entry_methods:
        if hasattr(optimizer_inst, method_name):
            method = getattr(optimizer_inst, method_name)
            sig_m = inspect.signature(method)
            num_params = len(sig_m.parameters)
            
            if num_params >= 2:
                best_f, best_x_normalized = method(bounded_func, bounded_grad)
            else:
                best_f, best_x_normalized = method(bounded_func)
            executed = True
            break
            
    if not executed:
        print("Error: No valid execution entry method found on the optimizer class.")
        sys.exit(1)
        
    end_time = time.time()
    
    # 7. Map back to real space
    best_x_real = lb + (best_x_normalized + 1.0) / 2.0 * (ub - lb)
    best_x_real = np.clip(best_x_real, lb, ub)
    
    print(f"\nOptimization Complete in {end_time - start_time:.2f}s")
    print(f"Best Normalized Loss Found: {best_f:.6f}")
    
    # 8. Visualization
    print("Generating Double-Gauss lens visualization...")
    fig, ax, final_loss = obj.visualize(theta=best_x_real, use_latex=False)
    plt.title(f"Optimized Double-Gauss - ID {opt_id[:8]} (Loss: {final_loss:.6f})")
    
    lens_visualisation_results = os.path.join(PROJECT_ROOT, "lens_visualisation_results")
    if not os.path.exists(lens_visualisation_results):
        os.makedirs(lens_visualisation_results)
        
    output_file = os.path.join(lens_visualisation_results, f"optimized_lens_aligned_{time.strftime('%H_%M_%d-%m')}.png")
    plt.savefig(output_file)
    print(f"Visualization saved successfully to: {output_file}")
    
    # Print out glass configurations for inspection
    x_cont, x_mat = obj.split_theta(best_x_real)
    print(f"\nOptimal Glass IDs found: {list(x_mat)}")

if __name__ == "__main__":
    main()
