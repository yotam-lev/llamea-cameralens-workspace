import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.prev_best_f = float('inf')
        self.last_eval_best = 0
        self.improve_rate = 0.1
        
        self.cond_num = 1.0
        self.H_reg = None
        self.Q = None
        self.eigs_vals = None
        self.flat_dir = None

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
        pop_size = 50
        pop = np.random.uniform(-1, 1, size=(pop_size, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])

        temp_max = 0.7
        levy_beta = 1.5
        base_exploit_freq = 6
        base_hess_freq = 12
        base_barrier_cond = 800

        while self.evals < self.budget:
            # --- Self-Tuning & Adaptive Mechanisms ---
            remaining_ratio = 1.0 - self.evals / self.budget
            
            if self.evals > self.last_eval_best:
                delta = max(1e-8, self.prev_best_f - self.best_f)
                self.improve_rate = 0.85 * self.improve_rate + 0.15 * delta
                self.prev_best_f = self.best_f
                self.last_eval_best = self.evals
            
            update_adaptation = self.evals % 10 == 0
            
            if update_adaptation:
                # 1. Adaptive Population Size: Expand on stagnation, contract on progress
                pop_size = int(np.clip(pop_size * 0.7 + 35 * (1.0 - self.improve_rate * 4), 25, 90))
                if pop_size > len(pop):
                    add = pop_size - len(pop)
                    new_p = np.random.uniform(-1, 1, size=(add, self.dim))
                    pop = np.vstack([pop, new_p])
                    pop_f = np.hstack([pop_f, np.array([self._evaluate(x, func) for x in new_p])])
                elif pop_size < len(pop):
                    idx_rm = np.argsort(pop_f)[pop_size:]
                    pop = np.delete(pop, idx_rm, 0)
                    pop_f = np.delete(pop_f, idx_rm)
                    
                # 2. Dynamic Parameters
                exploit_freq = int(np.clip(base_exploit_freq + 5 * (1.0 / (self.cond_num + 1.0)) - 3 * self.improve_rate * 4, 3, 12))
                barrier_cond = np.clip(base_barrier_cond * (1.0 - self.improve_rate * 3) + 200 * self.cond_num / 1000, 300, 1500)
                temp = temp_max * (1.0 - self.evals / self.budget)**np.clip(0.5 + 2.5 * (1.0 - self.improve_rate * 5), 0.5, 3.0)
                hess_freq = max(4, int(np.clip(12 * (1.0 + self.cond_num / 2000 - self.improve_rate * 5), 4, 20)))

            # Hessian Analysis (Curvature-Aware)
            if hess_func is not None and self.evals % hess_freq == 0:
                x_probe = pop[np.argmin(pop_f)].copy()
                x_probe[18:24] = np.clip(np.round(x_probe[18:24]), 0, 5).astype(int)
                if self.evals >= self.budget: break
                H_raw = hess_func(x_probe)[:18, :18]
                eigs_vals, Q = np.linalg.eigh(H_raw)
                min_abs_eig = np.min(np.abs(eigs_vals))
                self.cond_num = np.max(np.abs(eigs_vals)) / (min_abs_eig + 1e-8)
                self.H_reg = Q @ np.diag(np.abs(eigs_vals) + 1e-6) @ Q.T
                self.Q = Q
                self.eigs_vals = eigs_vals
                flat_idx = np.argmin(np.abs(eigs_vals))
                self.flat_dir = Q[:, flat_idx]

            best_idx = np.argmin(pop_f)
            center_c = pop[best_idx][:18]
            
            # 1. Barrier-Crossing Levy-Propagation
            if self.evals < self.budget:
                u = np.random.randn(pop_size, 18)
                v = np.random.randn(pop_size, 18)
                levy_steps = u / (np.abs(v)**(1.0/levy_beta))
                levy_steps *= np.random.uniform(0.2, 0.8, (pop_size, 18))

                scale_factor = np.clip(1.0 / (self.cond_num + 1.0), 0.1, 1.0)
                
                if self.H_reg is not None:
                    step_dirs = self.Q @ (levy_steps.T / (np.abs(self.eigs_vals) + 1e-8)).T
                    step_dirs *= scale_factor
                else:
                    step_dirs = levy_steps * scale_factor

                if self.cond_num > barrier_cond and np.random.rand() < temp:
                    barrier_amp = np.random.exponential(0.5) * np.sqrt(self.cond_num / barrier_cond)
                    step_dirs += np.outer(np.random.choice([-1, 1], pop_size), self.flat_dir * barrier_amp)

                new_c = np.clip(center_c + step_dirs, -1.0, 1.0)
                new_cat = np.zeros((pop_size, 6), dtype=int)
                base_cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                mut_prob = np.clip(temp * 0.5, 0.1, 0.8)
                for i in range(pop_size):
                    if np.random.rand() < mut_prob:
                        mut = np.random.choice([-1, 0, 1], size=6)
                        new_cat[i] = np.clip(np.round(base_cat + mut), 0, 5).astype(int)
                    else:
                        new_cat[i] = base_cat.copy()

                new_pop = np.hstack([new_c, new_cat])
                eval_f = np.array([self._evaluate(np.hstack([new_c[i], new_cat[i]]), func) for i in range(pop_size)])
                for i in range(pop_size):
                    if eval_f[i] < pop_f[i]:
                        pop[i] = np.hstack([new_c[i], new_cat[i]])
                        pop_f[i] = eval_f[i]

            if self.evals >= self.budget: break
            # 2. Hessian-Regularized Trust-Region Exploitation
            if self.evals % exploit_freq == 0 and hess_func is not None:
                best_idx = np.argmin(pop_f)
                xc = pop[best_idx][:18]
                cat = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)

                def obj(xc_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc_sub, cat]))
                def jac(xc_sub):
                    if self.evals >= self.budget: return np.zeros(18)
                    if grad_func: return grad_func(np.concatenate([xc_sub, cat]))[:18]
                    return np.zeros(18)
                def hess_closure(xc_sub):
                    return self.H_reg if self.H_reg is not None else np.eye(18)

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