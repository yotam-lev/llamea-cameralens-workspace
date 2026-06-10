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
        x_clip = np.clip(x.copy(), -1.0, 1.0)
        x_clip[18:24] = np.clip(np.round(x_clip[18:24]), 0, 5).astype(int)
        f = func(x_clip)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x_clip.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        temp_max = 0.6
        exploit_freq = 5
        hess_freq = 8
        neg_eps = 1e-5

        while self.evals < self.budget:
            temp = temp_max * (1.0 - self.evals / self.budget)**0.5
            H_reg = None
            Q = None
            eigs_vals = None
            best_idx = np.argmin(pop_f)

            # Hessian Update with Saddle Detection & Adaptive Scaling
            if hess_func is not None and self.evals % hess_freq == 0:
                x_probe = pop[best_idx].copy()
                x_probe[18:24] = np.clip(np.round(x_probe[18:24]), 0, 5).astype(int)
                if self.evals >= self.budget: break
                H_raw = hess_func(x_probe)[:18, :18]
                eigs_vals, Q = np.linalg.eigh(H_raw)
                # Regularization: ensure positive-definiteness
                H_reg = Q @ np.diag(np.abs(eigs_vals) + 1e-6) @ Q.T
                # Check for saddle points (negative eigenvalues)
                self.has_saddle = np.any(eigs_vals < -neg_eps)

            # Mutation Phase
            for i in range(n_samples):
                if self.evals >= self.budget: break
                xc = pop[best_idx].copy()
                xc[18:24] = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
                
                # Continuous Mutation
                if H_reg is not None:
                    # Adaptive step size: inverse trace of absolute curvature
                    trace_abs = np.sum(np.abs(eigs_vals))
                    step_scale = 1.0 / (trace_abs / 18.0 + 1e-8)
                    
                    # Base Gaussian perturbation in scaled metric
                    mu_c = np.random.randn(18) * step_scale
                    
                    # Saddle Escaping Perturbation
                    mu_esc = np.zeros(18)
                    if self.has_saddle:
                        esc_idx = np.argmin(eigs_vals)
                        esc_dir = Q[:, esc_idx]
                        esc_strength = np.random.uniform(0.5, 1.5) * step_scale
                        mu_esc = esc_dir * esc_strength
                    
                    xc[18] = xc[18] + mu_c + mu_esc
                else:
                    xc = xc + np.random.randn(18) * temp * 0.3
                
                xc = np.clip(xc[:18], -1.0, 1.0)
                
                # Categorical Mutation
                cat = np.clip(np.round(xc[18:24]), 0, 5).astype(int)
                if np.random.rand() < temp * 0.3:
                    mut_cat = np.random.choice([-1, 0, 1], size=6)
                    cat = np.clip(cat + mut_cat, 0, 5).astype(int)
                
                new_x = np.concatenate([xc, cat.astype(float)])
                f = self._evaluate(new_x, func)
                if f < pop_f[i]:
                    pop[i] = new_x
                    pop_f[i] = f

            # Local Search Phase
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
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 25, 'verbose': 0})
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < pop_f[best_idx]:
                        pop[best_idx] = x_ref
                        pop_f[best_idx] = f_ref
                else:
                    break

        return self.best_f, self.best_x