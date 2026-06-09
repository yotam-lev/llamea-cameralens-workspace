import os
import numpy as np
from scipy.optimize import minimize

# Prevent pthread exhaustion errors by limiting underlying BLAS threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        # Strict boundary enforcement
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        # Categorical mapping
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def _regularize_hessian(self, H):
        eigvals = np.linalg.eigvalsh(H)
        shift = max(0.0, -eigvals.min() + 1e-4)
        return H + (shift + 1.0) * np.eye(H.shape[0])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        fitness = np.empty(n_samples)
        
        for i in range(n_samples):
            if self.evals >= self.budget: break
            fitness[i] = self._evaluate(pop[i], func)
            
        F = 0.5
        CR = 0.9
        iter_count = 0
        
        while self.evals < self.budget:
            iter_count += 1
            new_pop = np.empty_like(pop)
            new_fit = np.empty(n_samples)
            
            for i in range(n_samples):
                if self.evals >= self.budget: break
                
                r1, r2, r3 = np.random.choice(n_samples, 3, replace=False)
                while r1 == i: r1 = np.random.randint(n_samples)
                while r2 == i: r2 = np.random.randint(n_samples)
                while r3 == i: r3 = np.random.randint(n_samples)
                
                trial = pop[i].copy()
                
                # Mixed Mutation: Continuous perturbation scaled by categorical diversity
                cat_diversity = np.std(pop[:, 18:24])
                mut_scale = 1.0 + 0.2 * cat_diversity
                mut = pop[r1][:18] + F * mut_scale * (pop[r2][:18] - pop[r3][:18])
                trial[:18] += mut
                
                # Categorical mutation with adaptive step
                cat_idx = np.random.randint(6)
                trial[18 + cat_idx] = np.clip(trial[18 + cat_idx] + np.random.choice([-1, 1]), 0, 5)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, trial, pop[i])
                
                f = self._evaluate(trial, func)
                new_pop[i] = trial
                new_fit[i] = f
                
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial

            pop = new_pop
            fitness = new_fit
            
            # Conditional Category-Geometry Refinement
            if iter_count % 4 == 0:
                elite_indices = np.argsort(fitness)[:4]
                for idx in elite_indices:
                    if self.evals >= self.budget: break
                    
                    # 1. Continuous Refinement
                    x_c = pop[idx][:18].copy()
                    cat_fixed = pop[idx][18:24].copy()
                    
                    def make_sub_funcs(c_fixed):
                        def sub_func(x_c_val):
                            full_x = np.empty(self.dim)
                            full_x[:18] = x_c_val
                            full_x[18:24] = c_fixed
                            return self._evaluate(full_x, func)
                        def sub_grad(x_c_val):
                            if grad_func is None: return np.zeros(18)
                            full_x = np.empty(self.dim)
                            full_x[:18] = x_c_val
                            full_x[18:24] = c_fixed
                            return grad_func(full_x)[:18]
                        def sub_hess(x_c_val):
                            if hess_func is None: return np.eye(18)
                            full_x = np.empty(self.dim)
                            full_x[:18] = x_c_val
                            full_x[18:24] = c_fixed
                            return self._regularize_hessian(hess_func(full_x))
                        return sub_func, sub_grad, sub_hess

                    if self.evals >= self.budget: break
                    sub_func, sub_grad, sub_hess = make_sub_funcs(cat_fixed)
                    bounds = [(-1.0, 1.0) for _ in range(18)]
                    
                    res = minimize(sub_func, x_c, jac=sub_grad, hess=sub_hess, 
                                   method='trust-constr', bounds=bounds, 
                                   options={'maxiter': 20, 'maxfun': 50})
                    
                    if res.success or res.fun < fitness[idx]:
                        x_c_opt = res.x
                        f_cont = res.fun
                        
                        # 2. Category Sweep: Use optimized geometry to guide discrete choices
                        best_cat = cat_fixed.copy()
                        f_best_cat = f_cont
                        
                        # Check neighbors in category space
                        for c_dim in range(6):
                            for delta in [-1, 1]:
                                trial_cat = cat_fixed.copy()
                                trial_cat[c_dim] = np.clip(trial_cat[c_dim] + delta, 0, 5)
                                
                                # Quick geometry re-opt for this category
                                def quick_func(x_c_val, c=trial_cat):
                                    full_x = np.empty(self.dim)
                                    full_x[:18] = x_c_val
                                    full_x[18:24] = c
                                    return self._evaluate(full_x, func)
                                
                                res_cat = minimize(quick_func, x_c_opt, method='L-BFGS-B', 
                                                   bounds=bounds, options={'maxiter': 5, 'maxfun': 10})
                                f_swapped = res_cat.fun
                                
                                if f_swapped < f_best_cat:
                                    f_best_cat = f_swapped
                                    best_cat = trial_cat
                                    x_c_opt = res_cat.x
                                    
                        final_x = np.empty(self.dim)
                        final_x[:18] = x_c_opt
                        final_x[18:24] = best_cat
                        f_refined = self._evaluate(final_x, func)
                        
                        if f_refined < fitness[idx]:
                            fitness[idx] = f_refined
                            pop[idx] = final_x

            F = np.clip(F * 1.02, 0.4, 1.0)
            CR = np.clip(CR + 0.005, 0.5, 0.95)
            
        return self.best_f, self.best_x