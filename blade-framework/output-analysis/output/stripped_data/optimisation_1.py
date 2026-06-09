import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

    def _evaluate(self, x, func):
        if self.evals >= self.budget: return float('inf')
        # Strict Boundary and Casting Enforcement
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _regularize(self, H):
        # Ensure positive-definite by taking absolute eigenvalues
        eigs, Q = np.linalg.eigh(H)
        return Q @ np.diag(np.abs(eigs)) @ Q.T

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize population using standard numpy
        pop = np.random.uniform(-1, 1, size=(20, self.dim))
        pop_f = []
        for x in pop:
            if self.evals >= self.budget: break
            f = self._evaluate(x, func)
            pop_f.append(f)
        pop_f = np.array(pop_f)

        while self.evals < self.budget:
            # Memetic Second-Order Local Search on current best
            if hess_func is not None and grad_func is not None:
                best_idx = np.argmin(pop_f)
                x_best = pop[best_idx]
                cat_ids = x_best[18:24].copy()
                x_c = x_best[:18].copy()
                
                if self.evals < self.budget:
                    H_raw = hess_func(np.concatenate([x_c, cat_ids]))
                    H_reg = self._regularize(H_raw)
                    
                    def obj(xc): return func(np.concatenate([xc, cat_ids]))
                    def jac(xc): return grad_func(np.concatenate([xc, cat_ids]))
                    def hess(xc): return H_reg
                    
                    if self.evals < self.budget:
                        res = minimize(
                            obj, x_c,
                            jac=jac, hess=hess,
                            method='trust-constr',
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20}
                        )
                        if self.evals < self.budget:
                            new_x = np.concatenate([res.x, cat_ids])
                            new_f = self._evaluate(new_x, func)
                            pop[best_idx] = new_x
                            pop_f[best_idx] = new_f

            # Mixed-Variable Evolutionary Step
            p_idx = np.argsort(pop_f)[:10]
            parents = pop[p_idx]
            
            offspring = []
            for _ in range(20):
                mu = (parents[0] + parents[1] + parents[2] + parents[3]) / 4.0
                mu[0:18] += np.random.normal(0, 0.12, 18)
                mu[18:24] = np.clip(np.round(np.random.uniform(0.0, 5.99, 6)), 0, 5).astype(int)
                offspring.append(mu)
            offspring = np.array(offspring)
            
            pop_f_off = []
            for x in offspring:
                if self.evals >= self.budget: break
                f = self._evaluate(x, func)
                pop_f_off.append(f)
            pop_f_off = np.array(pop_f_off)
            
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, pop_f_off])
            keep = np.argsort(combined_f)[:20]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x
