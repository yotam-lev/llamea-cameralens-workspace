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
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        if self.evals >= self.budget:
            return float('inf')
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 45
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        stagnation_count = 0
        history = [np.min(pop_f)]

        while self.evals < self.budget:
            curr_min = np.min(pop_f)
            history.append(curr_min)
            if len(history) > 1 and history[-1] >= history[-2] - 1e-9:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Phase 1: Spectral Escape Sampling (curvature-guided manifold jumping)
            if stagnation_count >= 10 and hess_func is not None:
                best_idx = np.argmin(pop_f)
                xc, cat = pop[best_idx][:18], pop[best_idx][18:24]
                full_x = np.concatenate([xc, cat])
                
                if self.evals >= self.budget: break
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                
                # Compute spectral gap to determine escape scale
                max_eig = np.max(np.abs(eigs))
                min_eig = np.min(np.abs(eigs)) + 1e-8
                cond = max_eig / min_eig
                # Inverse-conditioning amplifies steps along flat directions
                escape_scale = np.clip(0.3 * (cond / (cond + 3.0)), 0.05, 0.6)
                
                # Perturb along lowest-curvature eigenmode
                low_dir = Q[:, np.argmin(np.abs(eigs))]
                delta = escape_scale * low_dir * np.random.uniform(-1, 1)
                new_xc = np.clip(xc + delta, -1.0, 1.0)
                
                # Adaptive categorical recombination during escape
                p_mut = np.clip(0.15 * (cond / (cond + 3.0)) + 0.2, 0.05, 0.5)
                new_cat = cat.copy()
                for k in range(6):
                    if np.random.rand() < p_mut:
                        new_cat[k] = np.clip(int(cat[k]) + np.random.choice([-1, 1]), 0, 5)
                    
                candidate = np.concatenate([new_xc, new_cat])
                f_new = self._evaluate(candidate, func)
                pop[best_idx] = candidate
                pop_f[best_idx] = f_new
                stagnation_count = 0
                continue

            # Phase 2: Hessian-Regularized Trust-Region Exploitation
            if hess_func is not None and self.evals % 6 == 0:
                best_idx = np.argmin(pop_f)
                xc, cat = pop[best_idx][:18], pop[best_idx][18:24]
                full_x = np.concatenate([xc, cat])
                
                if self.evals >= self.budget: break
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                # Ensure positive-definiteness via absolute eigenvalue projection
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-8) @ Q.T
                
                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat]))
                def jac(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func: return grad_func(np.concatenate([xc_sub, cat]))[:18]
                    return np.zeros(18)
                def hess(xc_sub):
                    if self.evals >= self.budget: return H_reg
                    return H_reg

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)] * 18, options={'maxiter': 25, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    pop[best_idx] = x_ref
                    pop_f[best_idx] = f_ref
                else:
                    break

            # Phase 3: Curvature-Adaptive Differential Evolution
            if self.evals < self.budget:
                idx_rand = np.random.choice(n_samples - 1, 3, replace=False)
                a, b, c = pop[idx_rand[0]], pop[idx_rand[1]], pop[idx_rand[2]]
                
                # Dynamic crossover factor based on local curvature stiffness
                if hess_func and self.evals % 12 == 0:
                    full_c = np.concatenate([c[:18], c[18:24]])
                    if self.evals >= self.budget: break
                    H_c = hess_func(full_c)
                    ec = np.linalg.eigvalsh(H_c)
                    cfactor = np.clip(1.0 / (np.max(np.abs(ec)) + 0.4), 0.4, 1.6)
                else:
                    cfactor = 1.0
                
                mu = a[:18] + cfactor * (b[:18] - c[:18])
                mu = np.clip(mu, -1.0, 1.0)
                
                # Categorical block crossover
                new_cat = np.zeros(6)
                for k in range(6):
                    new_cat[k] = a[18+k] if np.random.rand() < 0.5 else b[18+k]
                new_cat = np.clip(np.round(new_cat), 0, 5).astype(int)
                
                candidate = np.concatenate([mu, new_cat])
                f_new = self._evaluate(candidate, func)
                worst_idx = np.argmax(pop_f)
                pop[worst_idx] = candidate
                pop_f[worst_idx] = f_new

        return self.best_f, self.best_x