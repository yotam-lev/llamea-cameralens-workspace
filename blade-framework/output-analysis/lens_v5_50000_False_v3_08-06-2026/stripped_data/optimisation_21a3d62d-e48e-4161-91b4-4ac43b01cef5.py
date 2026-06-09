import numpy as n
from scipy.optimize import minimize

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
        eval_x = np.clip(x.copy(), -1.0, 1.0)
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
                
                # DE Mutation for continuous
                mut = pop[r1][:18] + F * (pop[r2][:18] - pop[r3][:18])
                trial[:18] += mut
                
                # Categorical mutation/swap
                cat_idx = np.random.randint(6)
                trial[18 + cat_idx] = np.random.randint(0, 6)
                
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
            
            # Periodic Second-Order Trust Region Refinement
            if iter_count % 5 == 0:
                elite_indices = np.argsort(fitness)[:5]
                for idx in elite_indices:
                    if self.evals >= self.budget: break
                    x_c = pop[idx][:18].copy()
                    cat_fixed = pop[idx][18:24].copy()
                    
                    def sub_func(x_c_val, c_fixed=cat_fixed):
                        full_x = np.empty(self.dim)
                        full_x[:18] = x_c_val
                        full_x[18:24] = c_fixed
                        return self._evaluate(full_x, func)
                        
                    def sub_grad(x_c_val, c_fixed=cat_fixed):
                        if grad_func is None: return np.zeros(18)
                        full_x = np.empty(self.dim)
                        full_x[:18] = x_c_val
                        full_x[18:24] = c_fixed
                        return grad_func(full_x)[:18]
                        
                    def sub_hess(x_c_val, c_fixed=cat_fixed):
                        if hess_func is None: return np.eye(18)
                        full_x = np.empty(self.dim)
                        full_x[:18] = x_c_val
                        full_x[18:24] = c_fixed
                        H = hess_func(full_x)
                        return self._regularize_hessian(H)

                    bounds = [(-1.0, 1.0) for _ in range(18)]
                    
                    if self.evals >= self.budget: break
                    
                    res = minimize(sub_func, x_c, jac=sub_grad, hess=sub_hess, 
                                   method='trust-constr', bounds=bounds, options={'maxiter': 50})
                    
                    if res.success or res.fun < fitness[idx]:
                        final_x = np.empty(self.dim)
                        final_x[:18] = res.x
                        final_x[18:24] = cat_fixed
                        f_refined = self._evaluate(final_x, func)
                        if f_refined < fitness[idx]:
                            fitness[idx] = f_refined
                            pop[idx] = final_x

            F = np.clip(F * 1.05, 0.4, 1.0)
            CR = np.clip(CR + 0.01, 0.5, 0.95)
            
        return self.best_f, self.best_x