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

        T_max = 0.4
        resonance_period = 25
        exploit_freq = 8

        while self.evals < self.budget:
            current_temp = T_max * (1.0 - self.evals / self.budget) ** 0.6

            # Phase 1: Curvature-Adaptive Stochastic Resonance Exploration
            if hess_func is not None and (self.evals % resonance_period == 0 or np.all(pop_f[1:] >= np.min(pop_f) - 1e-6)):
                best_idx = np.argmin(pop_f)
                x_best = pop[best_idx]
                full_x = np.concatenate([x_best[:18], x_best[18:24]])
                
                if self.evals >= self.budget: break
                H_raw = hess_func(full_x)[:18, :18]
                eigs, Q = np.linalg.eigh(H_raw)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(np.abs(eigs) + 1e-8))
                
                # Anisotropic noise scaled by inverse curvature, modulated for resonance escape
                noise = Q @ D_inv_sqrt @ np.random.randn(n_samples, 18)
                resonance = np.sin(2 * np.pi * self.evals / resonance_period)
                step_c = noise * current_temp * (1.0 + 0.5 * resonance)
                
                # Discrete mutation probability inversely proportional to curvature spread
                curv_spread = np.max(np.abs(eigs)) / (np.min(np.abs(eigs)) + 1e-8)
                p_cat_mutate = np.clip(1.0 / np.log10(curv_spread + 2.0), 0.05, 0.5)
                
                new_pop = np.hstack([x_best[:18] + step_c, np.zeros((n_samples, 6))])
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    if np.random.rand() < p_cat_mutate:
                        mut = np.random.choice([-1, 0, 1], size=6)
                        new_pop[i, 18:24] = np.clip(np.round(x_best[18:24] + mut), 0, 5).astype(int)
                    else:
                        new_pop[i, 18:24] = x_best[18:24].copy()
                        
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    f = self._evaluate(new_pop[i], func)
                    if f < pop_f[i]:
                        pop[i] = new_pop[i]
                        pop_f[i] = f
                continue

            # Phase 2: Hessian-Regularized Trust-Region Exploitation
            if self.evals % exploit_freq == 0 and hess_func is not None:
                best_idx = np.argmin(pop_f)
                xc, cat = pop[best_idx][:18], pop[best_idx][18:24]
                
                full_x = np.concatenate([xc, cat])
                H_raw = hess_func(full_x)[:18, :18]
                eigs, Q = np.linalg.eigh(H_raw)
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                
                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat]))
                def jac(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func is not None:
                        return grad_func(np.concatenate([xc_sub, cat]))[:18]
                    return np.zeros(18)
                def hess(xc_sub):
                    if self.evals >= self.budget: return H_reg
                    return H_reg

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)] * 18,
                               options={'maxiter': 20, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    pop[best_idx] = x_ref
                    pop_f[best_idx] = f_ref
                else:
                    break

            # Population Renewal via Geometry-Guided Differential Evolution
            worst_idx = np.argmax(pop_f)
            if self.evals < self.budget:
                idx_rand = np.random.choice(pop_size - 1, 3, replace=False)
                mu = pop[idx_rand[0]] + 0.8 * (pop[idx_rand[1]] - pop[idx_rand[2]])
                mu[:18] += np.random.normal(0, 0.15, 18)
                mu[18:24] = np.clip(np.round(np.random.uniform(0.0, 5.99, 6)), 0, 5).astype(int)
                mu = np.clip(mu, -1.0, 1.0)
                mu[18:24] = np.clip(np.round(mu[18:24]), 0, 5).astype(int)
                f_new = self._evaluate(mu, func)
                pop[worst_idx] = mu
                pop_f[worst_idx] = f_new

        return self.best_f, self.best_x