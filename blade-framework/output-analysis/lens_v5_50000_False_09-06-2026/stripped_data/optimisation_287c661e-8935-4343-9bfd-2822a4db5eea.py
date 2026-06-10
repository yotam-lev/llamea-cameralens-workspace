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
        n_samples = 50
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        temp_max = 0.7
        levy_beta = 1.5
        exploit_freq = 6
        hess_freq = 12
        barrier_cond = 800

        while self.evals < self.budget:
            temp = temp_max * (1.0 - self.evals / self.budget)**0.5

            # Hessian analysis for curvature projection & barrier direction
            H_reg = None
            flat_dir = None
            eigs_vals = None
            cond_num = 1.0
            best_idx = np.argmin(pop_f)
            
            if hess_func is not None and self.evals % hess_freq == 0:
                x_probe = pop[best_idx].copy()
                x_probe[18:24] = np.clip(np.round(x_probe[18:24]), 0, 5).astype(int)
                if self.evals >= self.budget: break
                H_raw = hess_func(x_probe)[:18, :18]
                eigs_vals, Q = np.linalg.eigh(H_raw)
                cond_num = np.max(np.abs(eigs_vals)) / (np.min(np.abs(eigs_vals)) + 1e-8)
                H_reg = Q @ np.diag(np.abs(eigs_vals) + 1e-6) @ Q.T
                flat_idx = np.argmin(np.abs(eigs_vals))
                flat_dir = Q[:, flat_idx]

            # 1. Barrier-Crossing Levy-Propagation (Escapes Deep Minima)
            if self.evals < self.budget:
                # Levy flight generation (alpha=1.5 approximation)
                u = np.random.randn(n_samples, 18)
                v = np.random.randn(n_samples, 18)
                levy_steps = u / (np.abs(v)**(1.0/levy_beta))
                levy_steps *= np.random.uniform(0.2, 0.8, (n_samples, 18))

                # Curvature-adaptive scaling & projection
                if H_reg is not None:
                    step_dirs = Q @ (levy_steps.T / (np.abs(eigs_vals) + 1e-8)).T
                    step_norms = np.linalg.norm(step_dirs, axis=1, keepdims=True)
                    scale = np.clip(1.0 / step_norms, 0, 1) * np.random.uniform(0.1, 0.5, (n_samples, 1))
                    step_dirs *= scale
                else:
                    step_dirs = levy_steps

                # Barrier injection along shallowest eigenvector to cross narrow valleys
                if cond_num > barrier_cond and np.random.rand() < temp:
                    barrier_amp = np.random.exponential(0.5) * np.sqrt(cond_num / barrier_cond)
                    step_dirs += np.outer(np.random.choice([-1, 1], n_samples), flat_dir * barrier_amp)

                # Apply perturbations centered on current best
                center_c = pop[best_idx][:18]
                new_c = np.clip(center_c + step_dirs, -1.0, 1.0)

                # Categorical diffusion
                new_cat = np.zeros((n_samples, 6), dtype=int)
                base_cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                for i in range(n_samples):
                    if np.random.rand() < temp * 0.4:
                        mut = np.random.choice([-1, 0, 1], size=6)
                        new_cat[i] = np.clip(np.round(base_cat + mut), 0, 5).astype(int)
                    else:
                        new_cat[i] = base_cat.copy()

                new_pop = np.hstack([new_c, new_cat])
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    f = self._evaluate(new_pop[i], func)
                    if f < pop_f[i]:
                        pop[i] = new_pop[i]
                        pop_f[i] = f

            # 2. Hessian-Regularized Trust-Region Exploitation
            if self.evals % exploit_freq == 0 and hess_func is not None:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18]
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                if self.evals >= self.budget: break

                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat]))
                def jac(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func: return grad_func(np.concatenate([xc_sub, cat]))[:18]
                    return np.zeros(18)
                def hess_closure(xc_sub):
                    return H_reg

                res = minimize(obj, xc, jac=jac, hess=hess_closure, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 20, 'verbose': 0})
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref
                else:
                    break

        return self.best_f, self.best_x