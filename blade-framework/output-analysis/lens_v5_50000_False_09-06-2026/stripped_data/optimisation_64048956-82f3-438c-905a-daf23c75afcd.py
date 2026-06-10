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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        prev_best = self.best_f
        stagnation = 0

        while self.evals < self.budget:
            if self.best_f == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = self.best_f

            if self.evals >= self.budget: break

            top_idx = np.argsort(pop_f)[:12]
            for i in top_idx:
                if self.evals >= self.budget: break
                x = pop[i]
                xc = x[:18].copy()
                xd = x[18:24].copy()

                if hess_func is not None:
                    if self.evals >= self.budget: break
                    full_x = np.concatenate([xc, xd])
                    H_raw = hess_func(full_x)
                    eigs, Q = np.linalg.eigh(H_raw)
                    
                    # Strict PD regularization
                    H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                    
                    # NOVEL COUPLING: Material-conditioned metric tensor
                    # Discrete IDs dynamically scale continuous regularization stiffness
                    mat_scale = np.array([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])[xd]
                    H_cond = H_reg + np.diag(mat_scale * 1e-4)

                    def obj(xc_in):
                        if self.evals >= self.budget: return float('inf')
                        return func(np.concatenate([xc_in, xd]))
                    def jac(xc_in):
                        if self.evals >= self.budget: return np.zeros(18)
                        if grad_func: return grad_func(np.concatenate([xc_in, xd]))
                        return np.zeros(18)
                    def hess(xc_in):
                        return H_cond

                    res = minimize(obj, xc, jac=jac, hess=hess,
                                   method='trust-constr',
                                   bounds=[(-1.0, 1.0)] * 18,
                                   options={'maxiter': 20, 'verbose': 0})
                    if self.evals >= self.budget: break

                    x_new = np.concatenate([res.x, xd])
                    f_new = self._evaluate(x_new, func)
                    if self.evals >= self.budget: break
                    pop[i] = x_new
                    pop_f[i] = f_new
                    continue

                # NOVEL DISCRETE UPDATE: Curvature-guided categorical mutation
                low_curv_eig = Q[:, :3] if hess_func else np.eye(6)[:3]
                proj_disc = np.abs(low_curv_eig.T @ xd)
                disc_sens = 1.0 / (np.abs(proj_disc) + 1e-8)
                mut_probs = disc_sens / disc_sens.sum()

                for d in range(6):
                    if self.evals >= self.budget: break
                    if np.random.rand() < 0.15:
                        new_id = np.random.choice([0,1,2,3,4,5], p=mut_probs)
                        pop[i, 18+d] = new_id
                        pop_f[i] = self._evaluate(pop[i], func)
                        if self.evals >= self.budget: break

            if self.evals >= self.budget: break

            # Population replenishment with curvature-aware LHS
            offspring = np.random.uniform(-1, 1, size=(n_samples, self.dim))
            if hess_func:
                if self.evals >= self.budget: break
                H_raw = hess_func(np.concatenate([pop[0][:18], pop[0][18:24]]))
                eigs, Q = np.linalg.eigh(H_raw)
                low_mask = eigs < np.median(eigs)
                if np.any(low_mask):
                    v_low = Q[:, low_mask]
                    offspring[:, :18] += v_low @ (v_low.T @ np.random.normal(0, 0.3, size=(v_low.shape[1], n_samples)))

            offspring_f = [self._evaluate(x, func) for x in offspring]
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, offspring_f])
            keep = np.argsort(combined_f)[:n_samples]
            pop = combined[keep]
            pop_f = combined_f[keep]
            if self.evals >= self.budget: break

        return self.best_f, self.best_x