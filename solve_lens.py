import os
import sys
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

# 2. Define the Optimizer (PASTE YOUR EXTRACTED CODE HERE)
# ---------------------------------------------------------
import numpy as np
def lhs(n_samples, n_dim):
    """
    Basic Latin Hypercube Sampling generating values in [-1, 1].
    """
    result = np.empty((n_samples, n_dim))
    d = 1.0 / n_samples
    for i in range(n_dim):
        # Generate uniform samples for each interval
        result[:, i] = np.random.uniform(
            low=np.arange(n_samples) * d,
            high=(np.arange(n_samples) + 1) * d,
            size=n_samples
        )
        # Shuffle the samples for this dimension
        np.random.shuffle(result[:, i])
    
    # Map from [0, 1] to [-1, 1] to match the optimizer's bounds
    return result * 2.0 - 1.0

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.grad0_cont = None

    def set_initial_gradient(self, grad0_cont):
        self.grad0_cont = grad0_cont

    def _evaluate(self, x, func):
        """Wrapper to safely track budget and update best solution."""
        if self.evals >= self.budget:
            return float('inf')
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None):
        # Initialization (LHS)
        initial_population = lhs(n_samples=20, n_dim=self.dim)
        for x in initial_population:
            self._evaluate(x, func)

        # Initial gradient step
        if grad_func is not None and self.grad0_cont is not None:
            baseline_x = np.zeros(self.dim)
            lr = 0.01  # Learning rate for the gradient step
            gradient_step = lr * self.grad0_cont[:18]
            baseline_x[:18] -= gradient_step
            self._evaluate(baseline_x, func)

        # Enhanced Random Search with Gaussian mutation 
        while self.evals < self.budget:
            if np.random.rand() < 0.5:
                x = np.random.uniform(-1, 1, self.dim)
            else:
                # Gaussian mutation around the best found solution
                sigma = 0.1  # Mutation strength
                x = self.best_x.copy()
                x[:18] += np.random.normal(0, sigma, 18)  # Mutate continuous variables
                # Ensure bounds are respected for continuous variables
                x[:18] = np.clip(x[:18], -1, 1)
                # Categorical variables remain unchanged as they are discrete

            self._evaluate(x, func)

        return self.best_f, self.best_x
# ---------------------------------------------------------

def main():
    from examples.double_gauss_objective import DoubleGaussObjective
    
    # 3. Initialize Objective
    print("Initializing Double-Gauss Objective...")
    obj = DoubleGaussObjective(enable_grad=False, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta

    # Benchmark: Template Loss
    x_cont_init, x_mat_init = obj.init_from_templates()
    theta_init = obj.pack_theta(x_cont_init, x_mat_ init)
    loss_init = obj.objective_theta(theta_init)
    print(f"Initial Template Loss: {loss_init:.6f}")
    
    # We use a larger budget for a "production" run
    budget = 50000 
    seed = 25
    np.random.seed(seed)
    
    print(f"Running optimization (Budget: {budget}, Seed: {seed})...")
    optimizer = Optimizer(budget=budget, dim=dim)
    
    # Wrapper to handle normalization [-1, 1] -> [lb, ub]
    def bounded_func(x_normalized):
        x_real = lb + (x_normalized + 1.0) / 2.0 * (ub - lb)
        # Objective handles its own clipping internally usually, 
        # but we ensure it's within bounds.
        return obj.objective_theta(np.clip(x_real, lb, ub))

    import time
    start_time = time.time()
    best_f, best_x_normalized = optimizer(bounded_func)
    end_time = time.time()
    
    # 4. Map back to real space
    best_x_real = lb + (best_x_normalized + 1.0) / 2.0 * (ub - lb)
    best_x_real = np.clip(best_x_real, lb, ub)
    
    print(f"\nOptimization Complete in {end_time - start_time:.2f}s")
    print(f"Best Loss Found: {best_f:.6f}")
    
    # 5. Visualization
    print("Generating visualization...")
    fig, ax, final_loss = obj.visualize(theta=best_x_real, use_latex=False)
    plt.title(f"Optimized Double-Gauss (Loss: {final_loss:.6f})")
    

    lens_visualisation_results = os.path.join(PROJECT_ROOT, "lens_visualisation_results")
    if not os.path.exists(lens_visualisation_results):
        os.makedirs(lens_visualisation_results)
    output_file = os.path.join(lens_visualisation_results, f"optimized_lens{time.strftime('%H_%d-%m')}.png")
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
