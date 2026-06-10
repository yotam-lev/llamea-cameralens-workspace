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

    def _curvature_deflected_jump(self, x_c, cat_ids, hess_func):
        full_x = np.concatenate([x_c, cat_ids])
        H_raw = hess_func(full_x)
        eigs, Q = np.linalg.eigh(H_raw)
        
        # Anisotropic scaling: inversely proportional to sqrt(|lambda|) to stretch narrow valleys
        scale = np.abs(eigs) + 1e-8
        cond = np.max(scale) / np.min(scale)
        alpha = np.clip(np.log10(cond) * 0.15 + 0.3, 0.4, 2.0)
        D_inv = np.diag(1.0 / (scale ** (alpha * 0.5)))
        step = Q @ D_inv @ Q.T @ np.random.standard_cauchy(self.dim)
        
        # Deflection along strongest negative curvature to escape deep minima
        neg_mask = eigs < -1e-5
        if np.any(neg_mask):
            idx_neg = np.argmin(eigs)
            v_neg = Q[:, idx_neg]
            deflection_mag = np.abs(eigs[idx_neg]) * 0.4
            step -= deflection_mag * v_neg
            
        x_new_c = x_c + 1.8 * step
        return np.concatenate([x_new_c, cat_ids])

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 30
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = [self._evaluate(x, func) for x in pop]

        stagnation = 0
        prev_best = self.best_f

        while self.evals < self.budget:
            if self.best_f == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = self.best_f

            # Toggle between aggressive exploration and rapid exploitation
            if stagnation >= 12 and hess_func is not None:
                # Phase 1: Curvature-Deflected Levy Jump for basin escape
                new_pop = []
                new_f = []
                for x in pop:
                    if self.evals >= self.budget: break
                    x_new = self._curvature_deflected_jump(x[:18], x[18:24], hess_func)
                    f = self._evaluate(x_new, func)
                    new_pop.append(x_new)
                    new_f.append(f)
                if self.evals >= self.budget: break
                pop = np.array(new_pop)
                pop_f = np.array(new_f)
                stagnation = 0
                continue

            # Phase 2: Hessian-regularized Trust-Region Exploitation
            best_idx = np.argmin(pop_f)
            x_best = pop[best_idx]
            x_c, cat = x_best[:18], x_best[18:24]

            if hess_func is not None:
                full_x = np.concatenate([x_c, cat])
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                # Ensure positive-definite for solver stability
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                
                def obj(xc):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc, cat]))
                def jac(xc):
                    if self.evals >= self.budget: return np.zeros_like(xc)
                    return grad_func(np.concatenate([xc, cat]))
                def hess(xc):
                    return H_reg

                res = minimize(
                    obj, x_c, jac=jac, hess=hess,
                    method='trust-constr',
                    bounds=[(-1.0, 1.0)] * 18,
                    options={'maxiter': 15, 'verbose': 0}
                )
                if self.evals < self.budget:
                    x_refined = np.concatenate([res.x, cat])
                    f_refined = self._evaluate(x_refined, func)
                    pop[best_idx] = x_refined
                    pop_f[best_idx] = f_refined
                else:
                    break

            # Differential Evolution for population diversity
            p_idx = np.argsort(pop_f)[:15]
            offspring = []
            for _ in range(n_samples):
                idx_rand = np.random.choice(p_idx, 3, replace=False)
                mu = pop[idx_rand[0]] + 0.7 * (pop[idx_rand[1]] - pop[idx_rand[2]])
                mu[:18] += np.random.normal(0, 0.12, 18)
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
            keep = np.argsort(combined_f)[:n_samples]
            pop = combined[keep]
            pop_f = combined_f[keep]

        return self.best_f, self.best_x