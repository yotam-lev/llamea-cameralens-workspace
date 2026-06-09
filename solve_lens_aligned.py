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
    import re
    extracted_dir = os.path.join(PROJECT_ROOT, "extracted_gen_data")
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Directory {extracted_dir} does not exist.")
    
    # Recursively find all .json files
    files = []
    for root, _, filenames in os.walk(extracted_dir):
        for f in filenames:
            if f.endswith(".json"):
                files.append(os.path.relpath(os.path.join(root, f), extracted_dir))
    
    # Natural sort key (e.g. gen_2 before gen_10)
    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
    files = sorted(files, key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {extracted_dir} or its subfolders.")
        
    if json_filename:
        # Match by filename or partial name
        matched = [f for f in files if json_filename in f]
        if matched:
            target_file = os.path.join(extracted_dir, matched[0])
        else:
            print(f"\nError: Could not find file matching '{json_filename}' in {extracted_dir} or subfolders.")
            print("Available files (first 20 sorted naturally):")
            for f in files[:20]:
                print(f"  {f}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more files.")
            raise FileNotFoundError(f"Could not find file matching '{json_filename}'.")
    else:
        # Interactive selection if in a terminal/TTY
        if sys.stdin.isatty():
            print("\n" + "=" * 60)
            print(" INTERACTIVE OPTIMIZER SELECTION ")
            print("=" * 60)
            total = len(files)
            if total <= 25:
                for idx, f in enumerate(files):
                    print(f"  [{idx + 1:<2}] {f}")
            else:
                print("First 5 (early generations):")
                for idx in range(5):
                    print(f"  [{idx + 1:<2}] {files[idx]}")
                print("  ...")
                print("Last 15 (latest/evolved generations):")
                for idx in range(total - 15, total):
                    print(f"  [{idx + 1:<2}] {files[idx]}")
            
            print(f"\nSelect an optimizer number (1-{total}), or a unique substring, or press Enter for default (first generation): ", end="", flush=True)
            choice = sys.stdin.readline().strip()
            
            if not choice:
                target_file = os.path.join(extracted_dir, files[0])
            else:
                # Check if user entered an integer index
                try:
                    selected_idx = int(choice) - 1
                    if 0 <= selected_idx < total:
                        target_file = os.path.join(extracted_dir, files[selected_idx])
                    else:
                        print(f"Selection out of range. Defaulting to: {files[0]}")
                        target_file = os.path.join(extracted_dir, files[0])
                except ValueError:
                    # Treat as partial name matching
                    matched = [f for f in files if choice in f]
                    if matched:
                        target_file = os.path.join(extracted_dir, matched[0])
                    else:
                        print(f"No match for '{choice}'. Defaulting to: {files[0]}")
                        target_file = os.path.join(extracted_dir, files[0])
        else:
            # Default to the first one when not interactive
            target_file = os.path.join(extracted_dir, files[0])
        
    print(f"\nLoading optimizer code from: {os.path.basename(target_file)}")
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
    eval_count = 0
    best_loss_so_far = float('inf')
    eval_history = []  # Tuples of (eval_count, loss, best_loss_so_far)

    def bounded_func(xn):
        nonlocal eval_count, best_loss_so_far
        eval_count += 1
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        f_val = obj.objective_theta(np.clip(xr, lb, ub))
        
        if f_val < best_loss_so_far:
            best_loss_so_far = f_val
        eval_history.append((eval_count, f_val, best_loss_so_far))
        
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
    
    # --- Multi-Agent Orchestration: Reporter Agent ---
    print("\n" + "=" * 40)
    print(" BUDGET MILESTONES (EVALUATIONS) ")
    print("=" * 40)
    print(f"{'Budget Factor':<15} | {'Best Loss Achieved':<20}")
    print("-" * 40)
    
    milestones = [500, 1000, 5000, 10000, 25000, 50000]
    history_idx = 0
    total_evals = len(eval_history)
    
    for ms in milestones:
        if total_evals == 0:
            print(f"{ms:<15} | N/A")
            continue
            
        target_idx = min(ms, total_evals) - 1
        achieved_loss = eval_history[target_idx][2]
        
        if ms <= total_evals:
            print(f"{ms:<15} | {achieved_loss:.6f}")
        else:
            print(f"{ms:<15} | {achieved_loss:.6f} (Stopped at {total_evals})")
    
    print("=" * 40 + "\n")
    
    # --- Multi-Agent Orchestration: Visualization Agent ---
    print("Generating convergence graph...")
    if total_evals > 0:
        evals_arr = np.array([item[0] for item in eval_history])
        best_loss_arr = np.array([item[2] for item in eval_history])
        
        # Premium styling
        plt.style.use('dark_background')
        fig_conv, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        fig_conv.subplots_adjust(hspace=0.05)
        
        # Top subplot (Log Scale)
        ax_log.plot(evals_arr, best_loss_arr, color='#00d2ff', linewidth=2, label='Best Loss So Far')
        ax_log.set_yscale('log')
        ax_log.set_ylabel('Loss (Log Scale)', fontsize=12, color='white')
        ax_log.set_title(f'Convergence Graph - ID {opt_id[:8]}', fontsize=14, fontweight='bold', color='white')
        ax_log.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax_log.legend(facecolor='black', edgecolor='white')

        # Get max y for bottom plot from budget factor 500
        zoom_y_max = 2.0
        if total_evals > 0:
            target_idx = min(500, total_evals) - 1
            zoom_y_max = best_loss_arr[target_idx]
            zoom_y_max = max(zoom_y_max, 0.01) # fallback to avoid zero height
            
        # Bottom subplot (Linear Scale zoomed)
        ax_lin.plot(evals_arr, best_loss_arr, color='#00d2ff', linewidth=2)
        ax_lin.set_ylim(0, zoom_y_max * 1.05) # 5% padding
        ax_lin.set_xlabel('Number of Evaluations', fontsize=12, color='white')
        ax_lin.set_ylabel(f'Loss (Zoom [0, {zoom_y_max:.2f}])', fontsize=12, color='white')
        ax_lin.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add milestone markers
        for ms in milestones:
            if ms <= total_evals:
                ms_idx = ms - 1
                # Add to top (log) plot
                ax_log.plot(evals_arr[ms_idx], best_loss_arr[ms_idx], 'o', color='#ff007f', markersize=6)
                ax_log.annotate(f"{ms}", 
                                 (evals_arr[ms_idx], best_loss_arr[ms_idx]),
                                 textcoords="offset points", xytext=(0,10), ha='center',
                                 color='#ff007f', fontsize=9)
                # Add to bottom (linear) plot if within range
                if best_loss_arr[ms_idx] <= zoom_y_max * 1.1:
                    ax_lin.plot(evals_arr[ms_idx], best_loss_arr[ms_idx], 'o', color='#ff007f', markersize=6)
                    ax_lin.annotate(f"{ms}", 
                                     (evals_arr[ms_idx], best_loss_arr[ms_idx]),
                                     textcoords="offset points", xytext=(0,10), ha='center',
                                     color='#ff007f', fontsize=9)
        
        lens_visualisation_results = os.path.join(PROJECT_ROOT, "lens_visualisation_results")
        os.makedirs(lens_visualisation_results, exist_ok=True)
        conv_output_file = os.path.join(lens_visualisation_results, f"lens_convergence_{time.strftime('%H_%M_%d-%m')}.png")
        plt.savefig(conv_output_file, dpi=300, bbox_inches='tight')
        print(f"Convergence graph saved successfully to: {conv_output_file}\n")
        plt.close(fig_conv)
        plt.style.use('default')
    
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
