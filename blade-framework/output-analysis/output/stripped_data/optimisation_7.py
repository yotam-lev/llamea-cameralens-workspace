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
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x.copy(), -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _adaptive_step(self, x, func, grad_func, hess_func):
        x_c = x[:18]
        cat_ids = x[18:24]
        H_raw = hess_func(np.concatenate([x_c, cat_ids]))
        eigs, Q = np.linalg.eigh(H_raw)
        
        min_eig = eigs[0]
        max_eig = eigs[-1]
        eps = 1e-8
        scale = 0.8
        alpha_saddle = 0.5
        thresh_cond = 50.0
        
        # Saddle Point Escape
        if min_eig < -eps:
            idx_neg = np.argmin(eigs)
            v_neg = Q[:, idx_neg]
            x_new_c = x_c - alpha_saddle * v_neg
        # Anisotropic Basin / Deep Minimum Escape
        elif min_eig > eps and (max_eig / (min_eig + eps)) > thresh_cond:
            # Curvature-scaled Lévy flight
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.abs(eigs) + eps))
            levy = np.random.standard_cauchy(self.dim)
            x_new_c = x_c + scale * D_inv_sqrt @ levy
        # Standard Exploitation
        else:
            H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
            def obj(xc): return func(np.concatenate([xc, cat_ids]))
            def jac(xc): return grad_func(np.concatenate([xc, cat_ids]))
            def hess(xc): return H_reg
            
            res = minimize(
                obj, x_c, jac=jac, hess=hess,
                method='trust-constr',
                bounds=[(-1.0, 1.0)] * 18,
                options={'maxiter': 20, 'verbose': 0}
            )
            if self.evals < self.budget:
                x_new_c = res.x
            else:
                x_new_c = x_c # Abort if budget hit during sub-evals
            
        return np.concatenate([x_new_c, cat_ids])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize population using standard numpy
        pop = np.random.uniform(-1, 1, size=(25, self.dim))
        pop_f = []
        for x in pop:
            if self.evals >= self.budget: break
            f = self._evaluate(x, func)
            pop_f.append(f)
        pop_f = np.array(pop_f)

        stagnation_count = 0
        prev_best = self.best_f

        while self.evals < self.budget:
            if self.best_f == prev_best:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best = self.best_f

            # Trigger global reset if stuck
            if stagnation_count > 15:
                pop = np.random.uniform(-1, 1, size=(25, self.dim))
                pop_f = []
                for x in pop:
                    if self.evals >= self.budget: break
                    f = self._evaluate(x, func)
                    pop_f.append(f)
                pop_f = np.array(pop_f)
                stagnation_count = 0
                continue

            # Select worst individual to perturb
            worst_idx = np.argmax(pop_f)
            x_worst = pop[worst_idx]
            
            if hess_func is not None and grad_func is not None:
                # Apply HEAE step
                x_new = self._adaptive_step(x_worst, func, grad_func, hess_func)
                new_f = self._evaluate(x_new, func)
                pop[worst_idx] = x_new
                pop_f[worst_idx] = new_f

            # Differential Evolution for diversity
            p_idx = np.argsort(pop_f)[:12]
            offspring = []
            for _ in range(25):
                idx_rand = np.random.choice(p_idx, 3, replace=False)
                mu = pop[idx_rand[0]] + 0.8 * (pop[idx_rand[1]] - pop[idx_rand[2]])
                mu[:18] += np.random.normal(0, 0.1, 18)
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
            keep = np.argsort(combined_f)[:25]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x
