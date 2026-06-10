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
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        x_curr = np.random.uniform(-1, 1, self.dim)
        f_curr = self._evaluate(x_curr, func)

        T = 1.0
        T_min = 1e-5
        accept_window = []
        stagnation = 0
        base_step_c = 0.4
        base_step_cat = 0.3

        for _ in range(self.budget):
            if self.evals >= self.budget: break

            # Self-tuning: Hessian-conditioned scaling & adaptive cooling
            reg_eigs = None
            Q = None
            inv_sqrt_vec = None
            
            if hess_func is not None:
                if self.evals >= self.budget: break
                full_x = np.concatenate([x_curr[:18], x_curr[18:24].astype(int)])
                H_raw = hess_func(full_x)
                eigs, Q = np.linalg.eigh(H_raw)
                # Force positive definiteness per critical solver rule
                reg_eigs = np.abs(eigs) + 1e-6
                inv_sqrt_vec = 1.0 / np.sqrt(reg_eigs)
                cond = np.max(reg_eigs) / np.min(reg_eigs) + 1e-6
                
                # Dynamic step scaling inversely to curvature magnitude & condition number
                base_step_c = 0.5 / (np.mean(reg_eigs) + 1e-6) / np.sqrt(cond)
                base_step_cat = 0.2 / (reg_eigs[0] + 1.0)
            else:
                base_step_c *= 0.99
                base_step_cat *= 0.99

            # Generate candidate via anisotropic Hessian-mapped noise
            x_prop = x_curr[:18].copy()
            noise_c = np.random.randn(18)
            if hess_func is not None:
                step_c = (Q * inv_sqrt_vec) @ noise_c
            else:
                step_c = noise_c
            x_prop[:18] += step_c * base_step_c
            x_prop[18:24] = x_curr[18:24] + np.random.randn(6) * base_step_cat

            f_cand = self._evaluate(x_prop, func)
            if f_cand == float('inf'): break

            delta = f_cand - f_curr
            accept_window.append(1.0 if delta <= 0 else 0.0)
            
            if delta <= 0:
                x_curr = x_prop
                f_curr = f_cand
                stagnation = 0
                if f_curr < self.best_f:
                    self.best_f = f_curr
                    self.best_x = x_curr.copy()
            elif np.random.rand() < np.exp(-delta / max(T, 1e-12)):
                x_curr = x_prop
                f_curr = f_cand
                stagnation = 0
            else:
                stagnation += 1

            # Adaptive cooling driven by runtime feedback (acceptance rate)
            if len(accept_window) >= 40:
                acc_rate = np.mean(accept_window)
                if acc_rate > 0.5:
                    T *= 0.98  # Fast cooling when accepting too many
                elif acc_rate < 0.25:
                    T *= 1.05  # Slow cooling when trapped in shallow traps
                accept_window = []

            # Self-tuning memetic trigger: stagnation + budget velocity
            remaining = self.budget - self.evals
            if stagnation > 12 or (self.evals > 0 and self.evals % 30 == 0):
                stagnation = 0
                if remaining > 50 and hess_func is not None:
                    try:
                        bounds = [(-1.0, 1.0)] * 18
                        # Regularized Hessian for exact solver
                        H_safe = Q @ np.diag(reg_eigs) @ Q.T
                        def local_hess(x):
                            return H_safe
                        res = minimize(
                            func, x_curr[:18].copy(),
                            jac=grad_func,
                            hess=local_hess,
                            method='trust-constr',
                            bounds=bounds,
                            options={'maxiter': min(15, max(1, remaining // 15))}
                        )
                        x_curr[:18] = res.x
                        f_curr = self._evaluate(np.concatenate([x_curr[:18], x_curr[18:24]]), func)
                        if f_curr < self.best_f:
                            self.best_f = f_curr
                            self.best_x = x_curr.copy()
                    except Exception:
                        pass

            if T < T_min: T = T_min

        return self.best_f, self.best_x