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
class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.continuous_dim = 18
        self.categorical_dim = 6

    def latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        samples = np.zeros((n_samples, self.dim))
        for i in range(self.continuous_dim):
            samples[:, i] = np.random.uniform(-1, 1, n_samples)
        for i in range(self.categorical_dim):
            samples[:, self.continuous_dim + i] = np.random.randint(0, 100, n_samples)
        return samples

    def differential_evolution(self, func, pop_size=50, max_iter=100) -> tuple[float, np.ndarray]:
        bounds = [(-1, 1)] * self.continuous_dim + [(0, 99)] * self.categorical_dim
        population = self.latin_hypercube_sampling(pop_size)
        fitness = np.array([func(ind) for ind in population])
        
        F = np.random.uniform(0.5, 0.9)
        CR = np.random.uniform(0.7, 0.9)

        for _ in range(max_iter):
            new_population = []
            new_fitness = []
            for i in range(pop_size):
                idxs = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[idxs]
                
                mutant = np.zeros(self.dim)
                for j in range(self.continuous_dim):
                    if np.random.rand() < CR or j == i:
                        mutant[j] = a[j] + F * (b[j] - c[j])
                    else:
                        mutant[j] = population[i][j]
                
                for j in range(self.categorical_dim):
                    mutant[self.continuous_dim + j] = np.random.randint(0, 100)
                
                new_f = func(mutant)
                if new_f < fitness[i]:
                    new_population.append(mutant)
                    new_fitness.append(new_f)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
        
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]

    def local_refinement(self, func, x0, max_iter=50) -> tuple[float, np.ndarray]:
        # Custom local refinement strategy for handling categorical variables
        continuous_x = x0[:self.continuous_dim]
        categorical_x = x0[self.continuous_dim:]
        
        # Optimize continuous part using Nelder-Mead
        def continuous_func(continuous_part):
            new_x = np.concatenate((continuous_part, categorical_x))
            return func(new_x)
        
        from scipy.optimize import minimize
        res = minimize(continuous_func, continuous_x, method='Nelder-Mead', options={'maxiter': max_iter})
        best_continuous_x = res.x
        
        # Optimize categorical part by evaluating neighbors
        best_categorical_f = float('inf')
        for i in range(self.categorical_dim):
            for j in [-1, 0, 1]:
                new_categorical_x = categorical_x.copy()
                new_categorical_x[i] = (new_categorical_x[i] + j) % 100
                new_x = np.concatenate((best_continuous_x, new_categorical_x))
                f = func(new_x)
                if f < best_categorical_f:
                    best_categorical_f = f
                    best_categorical_x = new_categorical_x
        
        return best_categorical_f, np.concatenate((best_continuous_x, best_categorical_x))

    def __call__(self, func) -> tuple[float, np.ndarray]:
        best_f = float('inf')
        best_x = None
        
        # Global exploration phase
        for _ in range(10):
            f, x = self.differential_evolution(func)
            if f < best_f:
                best_f = f
                best_x = x
        
        # Local exploitation phase
        if best_f == float('inf'):
            return best_f, np.zeros(self.dim)
        
        for _ in range(5):
            local_best_f, local_best_x = self.local_refinement(func, best_x)
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x
        
        return best_f, best_x
# ---------------------------------------------------------

def main():
    from examples.double_gauss_objective import DoubleGaussObjective
    import inspect
    
    # 3. Initialize Objective
    print("Initializing Double-Gauss Objective...")
    obj = DoubleGaussObjective(enable_grad=True, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta

    # Benchmark: Template Loss
    x_cont_init, x_mat_init = obj.init_from_templates()
    theta_init = obj.pack_theta(x_cont_init, x_mat_init)
    loss_init = obj.objective_theta(theta_init)
    print(f"Initial Template Loss: {loss_init:.6f}")
    
    # === PHASE 1: EXPLOIT AUTOPSY (PROOF OF FABRICATION) ===
    print("\n--- EXPLOIT AUTOPSY (PROOF OF FABRICATION) ---")
    theta_exploit = theta_init.copy()
    # Inject continuous phantom glass IDs into indices 18-23
    theta_exploit[18:24] = [36.432, 48.789, 34.112, 34.901, 48.345, 48.876]
    
    loss_continuous = obj.objective_theta(np.clip(theta_exploit, lb, ub))
    loss_projected = obj.objective_theta(obj.project_theta(theta_exploit, lb=lb, ub=ub))
    
    print(f"Loss with Continuous Phantom Glass IDs (np.clip): {loss_continuous:.6f}")
    print(f"Loss with Strict MINLP Snapping (project_theta):  {loss_projected:.6f}")
    print(f"Discrepancy / Optimization Distortion:            {loss_projected - loss_continuous:.6f}")
    print("------------------------------------------------\n")
    
    # Compute baseline gradient for initialization
    grad0_cont = obj.gradient_cont_int(x_cont_init, x_mat_init)
    
    # We use a larger budget for a "production" run
    budget = 10000 
    seed = 1
    np.random.seed(seed)
    
    # Dynamic signature check for Optimizer constructor
    sig_init = inspect.signature(Optimizer.__init__)
    init_params = sig_init.parameters
    
    kwargs = {}
    if "budget" in init_params:
        kwargs["budget"] = budget
    if "dim" in init_params:
        kwargs["dim"] = dim
    if "grad0_cont" in init_params:
        kwargs["grad0_cont"] = grad0_cont
        
    print(f"Running optimization (Budget: {budget}, Seed: {seed})...")
    if kwargs:
        optimizer = Optimizer(**kwargs)
    else:
        # Fallback positional matching
        num_args = len(init_params) - 1
        if num_args >= 3:
            optimizer = Optimizer(budget, dim, grad0_cont)
        elif num_args == 2:
            optimizer = Optimizer(budget, dim)
        else:
            optimizer = Optimizer(budget)
    
    # Wrapper to handle normalization [-1, 1] -> [lb, ub]
    def bounded_func(x_normalized):
        x_real = lb + (x_normalized + 1.0) / 2.0 * (ub - lb)
        x_proj = obj.project_theta(x_real, lb=lb, ub=ub)
        

        
        return obj.objective_theta(x_proj)

    scale = (ub - lb) / 2.0

    def bounded_grad(x_normalized):
        x_real = lb + (x_normalized + 1.0) / 2.0 * (ub - lb)
        x_proj = obj.project_theta(x_real, lb=lb, ub=ub)
        xc, xi = obj.split_theta(x_proj)
        

        
        # Recreate scaling & apply to continuous gradients
        g_val = obj.gradient_cont_int(xc, xi) * scale[:18]
        return g_val

    import time
    start_time = time.time()
    
    # Dynamic signature check for Optimizer.__call__
    sig_call = inspect.signature(optimizer.__call__)
    call_params = sig_call.parameters
    
    if "grad_func" in call_params or len(call_params) >= 2:
        best_f, best_x_normalized = optimizer(bounded_func, bounded_grad)
    else:
        best_f, best_x_normalized = optimizer(bounded_func)
        
    end_time = time.time()
    
    # 4. Map back to real space
    best_x_real = lb + (best_x_normalized + 1.0) / 2.0 * (ub - lb)
    best_x_real = obj.project_theta(best_x_real, lb=lb, ub=ub)
    
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
