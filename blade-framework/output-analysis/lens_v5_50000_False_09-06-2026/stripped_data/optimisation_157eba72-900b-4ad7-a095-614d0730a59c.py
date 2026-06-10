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
        self.best_history = []
        self.stagnation = 0
        self.H_reg = None
        self.H_inv_diag = None
        self.last_cond = 1.0
        self.ls_freq = 8
        self.hess_freq = 15
        self.F = 0.5
        self.CR = 0.9

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

    def _update_hessian(self, x):
        H = self._hess_func(x)
        eigs = np.linalg.eigvalsh(H)
        shift = max(0, 1e-4 - eigs.min())
        self.H_reg = H + shift * np.eye(18)
        self.H_inv_diag = 1.0 / np.clip(np.abs(eigs), 1e-4, None)
        self.last_cond = np.max(eigs) / np.max(np.abs(eigs.min()), 1e-6)

    def _adapt(self, it):
        progress = self.evals / self.budget
        if len(self.best_history) > 0:
            imp = self.best_history[-1] - self.best_f
            self.best_history.append(self.best_f)
            if abs(imp) < 1e-5:
                self.stagnation += 1
            else:
                self.stagnation = 0

        if self.stagnation > 4:
            self.F = min(1.0, self.F * 1.15)
            self.CR = max(0.1, self.CR - 0.05)
            self.ls_freq = max(2, self.ls_freq // 2)
            if self.last_cond > 100:
                self.hess_freq = max(5, self.hess_freq // 2)
        elif progress > 0.6:
            self.F = max(0.1, self.F * 0.95)
            self.ls_freq = max(4, self.ls_freq // 2)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        n_pop = 20
        pop = np.random.uniform(-1, 1, size=(n_pop, self.dim))
        fitness = np.full(n_pop, np.inf)

        for i in range(n_pop):
            if self.evals >= self.budget: break
            fitness[i] = self._evaluate(pop[i], func)
        
        self.best_history.append(self.best_f)
        it = 0

        while self.evals < self.budget:
            it += 1
            self._adapt(it)

            # Adaptive Hessian update
            if self._hess_func and (it % self.hess_freq == 0 or self.H_reg is None):
                idx = np.argmin(fitness)
                if self.evals < self.budget:
                    self._update_hessian(pop[idx])
                    self.best_history.append(self.best_f)

            best_idx = np.argmin(fitness)
            g_x = pop[best_idx].copy()
            f_g = fitness[best_idx]

            for i in range(n_pop):
                if self.evals >= self.budget: break
                a, b, c = np.random.choice(n_pop, 3, replace=False)
                diff = pop[a] - pop[b]
                
                # Hessian-scaled mutation
                if self.H_inv_diag is not None:
                    diff[:18] *= np.sqrt(self.H_inv_diag[:18])
                
                mutant = g_x + self.F * diff
                j_mut = np.random.randint(0, self.dim)

                for j in range(self.dim):
                    if np.random.random() < self.CR or j == j_mut:
                        if j < 18:
                            mutant[j] = mutant[j]
                        else:
                            mutant[j] = np.clip(np.round(mutant[j]), 0, 5).astype(int)
                    else:
                        mutant[j] = pop[i][j]

                mutant[:18] = np.clip(mutant[:18], -1.0, 1.0)
                f_m = self._evaluate(mutant, func)

                if f_m < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = f_m

            # Adaptive local search on elite
            if self.evals < self.budget and it % self.ls_freq == 0:
                top_idx = np.argmin(fitness)
                c = pop[top_idx][:18].copy()
                cat = np.clip(np.round(pop[top_idx][18:24]), 0, 5).astype(int)
                
                if self.H_reg is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        c, method='trust-constr', hess=lambda xc: self.H_reg,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 40, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand, func)
                        if f_c < fitness[top_idx]:
                            pop[top_idx] = cand
                            fitness[top_idx] = f_c

        return self.best_f, self.best_x