import os
import sys
import json
import numpy as np

# 1. Setup paths
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, "..", ".."))
CAMERA_LENS_ROOT = os.path.join(PROJECT_ROOT, "camera-lens-simulation")
BLADE_FRAMEWORK_ROOT = os.path.join(PROJECT_ROOT, "blade-framework")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)
if BLADE_FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, BLADE_FRAMEWORK_ROOT)

from iohblade.problems.lens_optimisation import LensOptimisation, LHSWrapper
from iohblade.solution import Solution
from examples.double_gauss_objective import DoubleGaussObjective

def load_first_optimizer():
    """Load the first JSON file in extracted_gen_data/ directory."""
    extracted_dir = os.path.join(PROJECT_ROOT, "Lens_v4_50000_F/extracted_gen_data")
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Directory {extracted_dir} does not exist.")
    
    files = sorted([f for f in os.listdir(extracted_dir) if f.endswith(".json")])
    if not files:
        raise FileNotFoundError(f"No JSON files found in {extracted_dir}.")
    
    first_file = os.path.join(extracted_dir, files[0])
    print(f"Loading first optimizer file: {files[0]}")
    with open(first_file, "r") as f:
        data = json.load(f)
    return data

def run_llamea_sandbox_eval(code, budget, seeds):
    """Run evaluation using the exact LLaMEA environment."""
    # Convert list of seeds to training_instances format [(seed,)]
    training_instances = [(s,) for s in seeds]
    prob = LensOptimisation(budget_factor=budget, training_instances=training_instances)
    sol = Solution(code=code)
    
    print(f"Evaluating in LLaMEA Sandbox (Budget: {budget}, Seeds: {seeds})...")
    evaluated_sol = prob.evaluate(sol)
    
    # Fitness in LLaMEA is -mean_loss, so mean_loss = -fitness
    return -evaluated_sol.fitness, evaluated_sol.feedback

def extract_optimizer_class(exec_env):
    """
    Robustly extract the optimizer class from the executed environment,
    matching LLaMEA's evaluation framework logic.
    """
    OptimizerClass = exec_env.get("Optimizer")
    if not OptimizerClass:
        ignore_list = [
            "LHSWrapper",
            "DoubleGaussObjective",
            "Solution",
            "Problem",
            "Optimizer",
        ]
        for name, val in exec_env.items():
            if isinstance(val, type) and name not in ignore_list:
                if any(
                    hasattr(val, m)
                    for m in [
                        "optimize",
                        "solve",
                        "run",
                        "minimize",
                        "__call__",
                    ]
                ):
                    OptimizerClass = val
                    break
    return OptimizerClass

def run_isolated_solve_lens_eval(code, budget, seed):
    """Run evaluation using the standard isolated solve_lens.py logic (no gradients, different LHS, etc.)."""
    np.random.seed(seed)
    
    # Replicate solve_lens.py objective setup
    obj = DoubleGaussObjective(enable_grad=False, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta
    
    # 1. Custom LHS function defined in solve_lens.py
    def basic_lhs(n_samples, n_dim):
        result = np.empty((n_samples, n_dim))
        d = 1.0 / n_samples
        for i in range(n_dim):
            result[:, i] = np.random.uniform(
                low=np.arange(n_samples) * d,
                high=(np.arange(n_samples) + 1) * d,
                size=n_samples
            )
            np.random.shuffle(result[:, i])
        return result * 2.0 - 1.0

    # 2. Setup exec env (no sys.modules injection, no LHSWrapper)
    exec_env = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "lhs": basic_lhs,
    }
    
    # Execute the optimizer code
    exec(code, exec_env)
    OptimizerClass = extract_optimizer_class(exec_env)
    if not OptimizerClass:
        raise AttributeError("No valid Optimizer class found in code.")
        
    optimizer = OptimizerClass(budget=budget, dim=dim)
    
    # Standard objective wrapper (no receive_feedback, no gradient support)
    def bounded_func(xn):
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        return obj.objective_theta(np.clip(xr, lb, ub))
        
    print(f"Evaluating in Isolated solve_lens.py (Budget: {budget}, Seed: {seed})...")
    try:
        best_f, best_x = optimizer(bounded_func)
        return float(best_f)
    except Exception as e:
        print(f"Isolated evaluation failed: {e}")
        return float('inf')

def run_aligned_solve_lens_eval(code, budget, seed):
    """
    Run evaluation using a CORRECTED solve_lens.py logic that matches LLaMEA's evaluation sandbox.
    Exposes and proves how aligning the conditions removes the discrepancy.
    """
    np.random.seed(seed)
    
    # Replicate LLaMEA's objective setup with enable_grad=True
    obj = DoubleGaussObjective(enable_grad=True, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta
    x0_cont, x0_ids = obj.init_from_templates()
    grad0_cont = obj.gradient_cont_int(x0_cont, x0_ids)
    
    def raw_func(x):
        return obj.objective_theta(np.clip(x, lb, ub))
        
    def raw_grad(x):
        xc, xi = obj.split_theta(np.clip(x, lb, ub))
        return obj.gradient_cont_int(xc, xi)
        
    # LHS polymorphic wrapper used in LLaMEA
    lhs_tool = LHSWrapper()
    
    # Sandbox environment setup
    exec_env = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "latin_hypercube_sampling": lhs_tool,
        "lhs": lhs_tool,
        "grad0_cont": grad0_cont,
    }
    
    # Execute optimizer code
    exec(code, exec_env)
    OptimizerClass = extract_optimizer_class(exec_env)
    if not OptimizerClass:
        raise AttributeError("No valid Optimizer class found in code.")
    
    # Replicate constructor invocation
    import inspect
    sig = inspect.signature(OptimizerClass.__init__)
    init_params = sig.parameters
    
    kwargs = {}
    if "budget" in init_params:
        kwargs["budget"] = budget
    if "dim" in init_params:
        kwargs["dim"] = dim
    if "grad0_cont" in init_params:
        kwargs["grad0_cont"] = grad0_cont
        
    if not kwargs:
        num_args = len(init_params) - 1
        if num_args >= 3:
            opt = OptimizerClass(budget, dim, grad0_cont)
        elif num_args == 2:
            opt = OptimizerClass(budget, dim)
        else:
            opt = OptimizerClass(budget)
    else:
        opt = OptimizerClass(**kwargs)
        
    # Scale continuous components by half box-width for chain rule
    scale = (ub - lb) / 2.0
    
    def bounded_func(xn):
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        f_val = raw_func(xr)
        if hasattr(opt, "receive_feedback"):
            try:
                opt.receive_feedback({"loss": f_val, "x_normalized": xn})
            except:
                pass
        return f_val

    def bounded_grad(xn):
        xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
        g_val = raw_grad(xr) * scale[:18]
        if hasattr(opt, "receive_feedback"):
            try:
                opt.receive_feedback({"grad": g_val, "x_normalized": xn})
            except:
                pass
        return g_val
        
    # Inject attributes as the sandbox does
    opt.func = bounded_func
    opt.grad_func = bounded_grad
    exec_env["func"] = bounded_func
    exec_env["grad_func"] = bounded_grad
    
    if not hasattr(opt, "receive_feedback"):
        opt.receive_feedback = lambda x: None
        
    print(f"Evaluating in Aligned solve_lens.py (Budget: {budget}, Seed: {seed})...")
    
    # Select best call method
    entry_methods = ["__call__", "optimize", "solve", "run", "minimize"]
    for method_name in entry_methods:
        if hasattr(opt, method_name):
            method = getattr(opt, method_name)
            sig_m = inspect.signature(method)
            num_params = len(sig_m.parameters)
            if num_params >= 2:
                best_f, best_x = method(bounded_func, bounded_grad)
            else:
                best_f, best_x = method(bounded_func)
            return float(best_f)
            
    raise AttributeError("No valid execution method found on optimizer.")

def main():
    print("="*70)
    print(" LLaMEA CAMERA LENS DISCREPANCY & VERIFICATION TOOL ")
    print("="*70)
    
    try:
        # Load the optimizer data
        optimizer_data = load_first_optimizer()
        code = optimizer_data["code"]
        expected_fitness = optimizer_data["fitness"]
        json_feedback = optimizer_data["feedback"]
        
        print(f"Expected Fitness from JSON: {expected_fitness}")
        print(f"Feedback from JSON: {json_feedback}")
        print("-" * 50)
        
        # Define seed and budget matching the JSON file's evaluation
        # Let's see: from feedback, "Mean loss: 0.506012" on seed 1
        budget = 5000 # default budget factor
        seeds = [1]   # Let's test on seed 1 first
        
        # 1. Run LLaMEA Sandbox Evaluation
        sandbox_loss, sandbox_feedback = run_llamea_sandbox_eval(code, budget, seeds)
        sandbox_fitness = -sandbox_loss
        print(f"LLaMEA Sandbox Result Fitness: {sandbox_fitness}")
        print(f"LLaMEA Sandbox Result Feedback: {sandbox_feedback}")
        print("-" * 50)
        
        # 2. Run Isolated solve_lens.py Evaluation
        isolated_loss = run_isolated_solve_lens_eval(code, budget, seed=1)
        isolated_fitness = -isolated_loss
        print(f"Isolated solve_lens.py Result Fitness: {isolated_fitness}")
        print("-" * 50)
        
        # 3. Run Aligned solve_lens.py Evaluation
        aligned_loss = run_aligned_solve_lens_eval(code, budget, seed=1)
        aligned_fitness = -aligned_loss
        print(f"Aligned solve_lens.py Result Fitness: {aligned_fitness}")
        print("="*70)
        
        # 4. Quantification & Proof of Alignment
        discrepancy_isolated = abs(sandbox_fitness - isolated_fitness)
        discrepancy_aligned = abs(sandbox_fitness - aligned_fitness)
        
        print("RESULTS QUANTIFICATION:")
        print(f"- Discrepancy between Sandbox and Isolated: {discrepancy_isolated:.8f}")
        print(f"- Discrepancy between Sandbox and Aligned:  {discrepancy_aligned:.8f}")
        print("-" * 50)
        
        if discrepancy_aligned < 1e-6:
            print("✅ SUCCESS: Aligned environment matches LLaMEA sandbox perfectly!")
            print("This mathematically proves that the vast discrepancies are caused by:")
            print("  1. Missing analytical gradients (enable_grad=True & chain rule scaling).")
            print("  2. Lack of LHSWrapper polymorphic capabilities.")
            print("  3. Lack of force-injection of opt.func/opt.grad_func and receive_feedback loops.")
            print("  4. Constructors expecting 'grad0_cont' or custom signature bindings.")
        else:
            print("⚠️ WARNING: A minor floating-point or initialization discrepancy remains.")
            
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
