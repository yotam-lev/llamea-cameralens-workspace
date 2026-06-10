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
        self.V = None
        self.D = None
        self.D_abs = None
        self.stagnation_steps = 0
        self.ls_freq = 6
        self.hess_freq = 10
        self.F = 0.8
        self.CR = 0.9
        self.esc_attempts = 0
        self.max_esc = 3
        self.pop_size = 30
        self.history = []

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

    def _update_hessian_info(self, x):
        if self._hess_func is None or self.evals >= self.budget:
            return
        H = self._hess_func(x)
        eigs = np.linalg.eigvalsh(H)
        shift = max(0, 1e-4 - eigs.min())
        self.H_reg = H + shift * np.eye(18)
        self.V, self.D = np.linalg.eigh(H)
        self.D_abs = np.abs(self.D)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        fitness = np.full(self.pop_size, np.inf)

        for i in range(self.pop_size):
            if self.evals >= self.budget: break
            fitness[i] = self._evaluate(pop[i], func)

        self.history.append(self.best_f)
        it = 0

        while self.evals < self.budget:
            it += 1
            imp = self.history[-1] - self.best_f
            self.history.append(self.best_f)
            if abs(imp) < 1e-6:
                self.stagnation_steps += 1
            else:
                self.stagnation_steps = 0
                self.esc_attempts = 0

            # Adaptive Hessian update
            if self._hess_func and (it % self.hess_freq == 0 or self.H_reg is None):
                best_idx = np.argmin(fitness)
                if self.evals < self.budget:
                    self._update_hessian_info(pop[best_idx])
                    self.history.append(self.best_f)

            best_idx = np.argmin(fitness)
            g_x = pop[best_idx].copy()

            # Novel: Curvature-Whitened DE with Eigen-Projection Escape
            if self.stagnation_steps > 2:
                if self.V is not None and self.esc_attempts < self.max_esc:
                    min_idx = np.argmin(self.D_abs)
                    escape_dir = self.V[:, min_idx] * np.sqrt(np.clip(self.D_abs[min_idx], 1e-4, None))
                    step = np.random.uniform(0.5, 2.0)
                    pop[best_idx][:18] += step * escape_dir
                    pop[best_idx][18:24] = np.clip(np.round(pop[best_idx][18:24]), 0, 5).astype(int)
                    pop[best_idx][:18] = np.clip(pop[best_idx][:18], -1.0, 1.0)
                    f_esc = self._evaluate(pop[best_idx], func)
                    if f_esc < fitness[best_idx]:
                        fitness[best_idx] = f_esc
                        self.stagnation_steps = 0
                    self.esc_attempts += 1
                else:
                    levy = np.random.normal(0, 1, size=(self.pop_size, self.dim))
                    u = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
                    step = levy / np.abs(u)**(1/3)
                    pop += step * 0.1
                    pop = np.clip(pop, -1.0, 1.0)
                    pop[:, 18:24] = np.clip(np.round(pop[:, 18:24]), 0, 5).astype(int)
                    fitness = np.array([self._evaluate(p, func) for p in pop])
                    self.stagnation_steps = 0
                    self.esc_attempts = 0
                    if self._hess_func and self.evals < self.budget:
                        self._update_hessian_info(pop[np.argmin(fitness)])
                    continue

            for i in range(self.pop_size):
                if self.evals >= self.budget: break
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                diff = pop[a] - pop[b]

                # Curvature Whitening: rotate into eigenbasis, scale by inverse sqrt curvature, rotate back
                if self.V is not None:
                    diff_eig = self.V.T @ diff[:18]
                    scale = 1.0 / np.sqrt(np.clip(self.D_abs, 1e-4, None))
                    diff_eig = diff_eig * scale
                    diff[:18] = self.V @ diff_eig

                mutant = g_x[:18] + self.F * diff
                j_mut = np.random.randint(0, self.dim)
                trial = np.copy(pop[i])

                for j in range(self.dim):
                    if np.random.random() < self.CR or j == j_mut:
                        if j < 18:
                            trial[j] = np.clip(mutant[j], -1.0, 1.0)
                        else:
                            trial[j] = np.clip(np.round(mutant[j]), 0, 5).astype(int)
                f_m = self._evaluate(trial, func)
                if f_m < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_m

            # Adaptive local search on elite using exact Hessian
            if self.evals < self.budget and it % self.ls_freq == 0:
                top_idx = np.argmin(fitness)
                c = pop[top_idx][:18].copy()
                cat = np.clip(np.round(pop[top_idx][18:24]), 0, 5).astype(int)
                if self.H_reg is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        c, method='trust-constr', hess=lambda xc: self.H_reg,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 50, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand, func)
                        if f_c < fitness[top_idx]:
                            pop[top_idx] = cand
                            fitness[top_idx] = f_c
                            self.stagnation_steps = 0

        return self.best_f, self.best_x