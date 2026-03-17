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
    
    # 3. Initialize Objective
    print("Initializing Double-Gauss Objective...")
    obj = DoubleGaussObjective(enable_grad=False, enable_hessian=False)
    lb, ub = obj.bounds()
    dim = obj.n_theta

    # Benchmark: Template Loss
    x_cont_init, x_mat_init = obj.init_from_templates()
    theta_init = obj.pack_theta(x_cont_init, x_mat_init)
    loss_init = obj.objective_theta(theta_init)
    print(f"Initial Template Loss: {loss_init:.6f}")
    
    # We use a larger budget for a "production" run
    budget = 10000 
    seed = 42
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
