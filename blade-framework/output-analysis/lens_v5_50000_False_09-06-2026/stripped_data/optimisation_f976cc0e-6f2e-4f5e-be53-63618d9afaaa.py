import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self._hess_func = None
        self.H_reg = None
        self.H_eigs = None
        self.H_Q = None
        self.last_cond = 1.0
        self.H_inv_diag = None
        self.stagnation = 0
        self.mut_scale = 0.02
        self.ls_freq = 4
        self.hess_freq = 5
        self.best_history = []

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

    def _update_hessian(self, x_full):
        H = self._hess_func(x_full)
        eigs, Q = np.linalg.eigh(H)
        eigs = np.abs(eigs) + 1e-4
        self.H_eigs = eigs
        self.H_Q = Q
        self.H_reg = Q @ np.diag(eigs) @ Q.T
        H_inv = Q @ np.diag(1.0 / eigs) @ Q.T
        self.H_inv_diag = np.diag(H_inv)
        self.last_cond = np.max(eigs) / np.min(eigs + 1e-6)

    def _adapt(self):
        progress = self.evals / self.budget
        self.best_history.append(self.best_f)
        if len(self.best_history) > 1:
            rel_imp = (self.best_history[-2] - self.best_f) / (abs(self.best_history[-2]) + 1e-6)
            self.stagnation = max(0, self.stagnation + (1 if rel_imp < 1e-4 else -1))
        
        # Adaptive local search frequency: drastically reduces expensive minimize calls as condition number worsens or budget depletes
        curv_penalty = 1.0 + np.log(self.last_cond + 1e-6)
        self.ls_freq = int(1 + progress * 8 * curv_penalty + max(0, self.stagnation * 2))
        
        # Adaptive Hessian update: reduces eigen-decomposition calls near convergence or when curvature is stable
        self.hess_freq = max(3, int(15 * (1 - progress) + (1 if self.H_reg is None else 0)))
        
        # Mutation scale decay
        self.mut_scale = max(1e-3, 0.02 * (1 - 0.5 * progress))

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        n_samples = 16
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        self.pop = pop
        self.fitness = np.full(n_samples, float('inf'))
        self.best_history.append(self.best_f)

        for i in range(n_samples):
            if self.evals >= self.budget: break
            self.fitness[i] = self._evaluate(self.pop[i], func)

        it = 0
        while self.evals < self.budget:
            it += 1
            self._adapt()
            
            if self.evals >= self.budget: break

            best_idx = np.argmin(self.fitness)
            g_x = self.pop[best_idx].copy()
            g_f = self.fitness[best_idx]

            # Adaptive Hessian update
            if self._hess_func is not None and (it % self.hess_freq == 0 or self.H_reg is None):
                if self.evals < self.budget:
                    self._update_hessian(g_x)
                    self.best_history.append(self.best_f)

            # Categorical mutation
            cat_rate = 0.05 * (1 + min(2, self.stagnation))
            if self.H_inv_diag is not None:
                agg_curv = np.mean(1.0 / (self.H_eigs + 1e-6))
                if self.last_cond > 100 or agg_curv > 0.2:
                    cat_rate += 0.1
            for k in range(6):
                if np.random.random() < cat_rate:
                    g_x[18:24][k] = np.random.randint(0, 6)

            # Preconditioned step
            if grad_func is not None and self.evals < self.budget:
                g_full = grad_func(g_x)
                g_c = g_full[:18]
            else:
                g_c = np.zeros(18)

            if self.H_inv_diag is not None:
                damp = min(1.0, 50.0 / (self.last_cond + 1.0))
                # Apply preconditioner only to continuous subspace
                d_c = -self.H_inv_diag[:18] * (g_c + 1e-6)
                g_x[:18] += self.mut_scale * damp * d_c
            g_x[:18] = np.clip(g_x[:18], -1.0, 1.0)

            # Adaptive local search: frequency suppressed by _adapt logic to prevent timeout
            if self.evals < self.budget and it % max(1, int(self.ls_freq)) == 0:
                x_c = g_x[:18].copy()
                cat_ids = np.clip(np.round(g_x[18:24]), 0, 5).astype(int)
                def obj(xc): return func(np.concatenate([xc, cat_ids]))
                # Strict iteration cap to guarantee termination
                res = minimize(obj, x_c, hess=lambda xc: self.H_reg, method='trust-constr',
                               bounds=[(-1.0, 1.0)]*18, options={'maxiter': 10, 'verbose': 0})
                if self.evals < self.budget:
                    cand = np.concatenate([res.x, cat_ids])
                    cand_f = self._evaluate(cand, func)
                    if cand_f < g_f:
                        g_f = cand_f
                        g_x = cand
                        self.fitness[best_idx] = cand_f
                        self.best_f = cand_f
                        self.best_x = cand.copy()

            # Population update
            if self.evals < self.budget and it % 4 == 0:
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    noise = np.random.normal(0, self.mut_scale, self.dim)
                    if self.H_inv_diag is not None:
                        noise[:18] *= np.clip(np.sqrt(self.H_inv_diag[:18]), 0.05, 2.0)
                    self.pop[i] = g_x + noise
                    self.fitness[i] = self._evaluate(self.pop[i], func)
                g_x = self.pop[np.argmin(self.fitness)].copy()
                g_f = self.fitness[np.argmin(self.fitness)]

        return self.best_f, self.best_x