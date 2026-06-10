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
        
        # Controller state
        self.F = 0.7
        self.CR = 0.9
        self.n_pop = 15
        self.ls_freq = 5
        self.hess_freq = 10
        
        self.last_best_f = float('inf')
        self.improv_hist = []
        self.cond_hist = []
        self.stagnation = 0
        self.H_reg = None
        self.H_inv_diag = None
        
        # History for population management
        self.pop = None
        self.fitness = None

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
        self.cond = np.max(np.abs(eigs)) / max(np.min(np.abs(eigs)), 1e-6)

    def _adapt(self):
        imp = self.last_best_f - self.best_f
        self.last_best_f = self.best_f
        self.improv_hist.append(imp)
        if len(self.improv_hist) > 20:
            self.improv_hist.pop(0)
            
        if hasattr(self, 'cond'):
            self.cond_hist.append(self.cond)
            if len(self.cond_hist) > 10:
                self.cond_hist.pop(0)

        mean_imp = np.mean(self.improv_hist[-10:])
        mean_cond = np.mean(self.cond_hist[-5:]) if self.cond_hist else 1.0
        
        if len(self.improv_hist) > 5:
            recent_trend = self.improv_hist[-1] - self.improv_hist[-5]
        else:
            recent_trend = 0.0

        if recent_trend < -1e-6: # Regression
            self.stagnation += 1
        elif abs(recent_trend) < 1e-6: # Stagnation
            self.stagnation += 1
        else:
            self.stagnation = 0
            
        if self.stagnation > 3:
            self.F = min(1.2, self.F * 1.1)
            self.ls_freq = max(2, self.ls_freq // 2)
        else:
            self.F = max(0.1, self.F * 0.98)
            
        if self.stagnation > 5 and mean_cond > 50:
            self.n_pop = min(50, self.n_pop + 2)
        elif mean_imp > 1e-4 and self.n_pop > 10:
            self.n_pop = max(10, self.n_pop - 1)
            
        if mean_cond > 100:
            self.ls_freq = max(2, self.ls_freq - 1)
        elif mean_imp < 1e-5:
            self.ls_freq = max(3, self.ls_freq - 1)
            
        if self.pop is not None:
            current_n = len(self.fitness)
            if self.n_pop != current_n:
                if self.n_pop > current_n:
                    needed = self.n_pop - current_n
                    new_particles = np.random.uniform(-1, 1, size=(needed, self.dim))
                    self.pop = np.vstack([self.pop, new_particles])
                    self.fitness = np.concatenate([self.fitness, np.full(needed, np.inf)])
                else:
                    idx = np.argsort(self.fitness)[:self.n_pop]
                    self.pop = self.pop[idx]
                    self.fitness = self.fitness[idx]

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        self.n_pop = 15
        self.pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        self.fitness = np.full(self.n_pop, np.inf)

        for i in range(len(self.fitness)):
            if self.evals >= self.budget: break
            self.fitness[i] = self._evaluate(self.pop[i], func)
        
        self.last_best_f = self.best_f
        it = 0

        while self.evals < self.budget:
            it += 1
            self._adapt()

            if self._hess_func and (it % self.hess_freq == 0 or self.H_reg is None):
                idx = np.argmin(self.fitness)
                if self.evals < self.budget:
                    self._update_hessian(self.pop[idx])

            best_idx = np.argmin(self.fitness)
            g_x = self.pop[best_idx].copy()

            for i in range(len(self.fitness)):
                if self.evals >= self.budget: break
                a, b, c = np.random.choice(len(self.fitness), 3, replace=False)
                diff = self.pop[a] - self.pop[b]
                
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
                        mutant[j] = self.pop[i][j]

                mutant[:18] = np.clip(mutant[:18], -1.0, 1.0)
                f_m = self._evaluate(mutant, func)

                if f_m < self.fitness[i]:
                    self.pop[i] = mutant
                    self.fitness[i] = f_m

            if self.evals < self.budget and it % self.ls_freq == 0:
                top_idx = np.argmin(self.fitness)
                c = self.pop[top_idx][:18].copy()
                cat = np.clip(np.round(self.pop[top_idx][18:24]), 0, 5).astype(int)
                
                if self.H_reg is not None:
                    res = minimize(
                        lambda xc: func(np.concatenate([xc, cat])),
                        c, method='trust-constr', hess=lambda xc: self.H_reg,
                        bounds=[(-1.0, 1.0)]*18, options={'maxiter': 40, 'verbose': 0}
                    )
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat])
                        f_c = self._evaluate(cand, func)
                        if f_c < self.fitness[top_idx]:
                            self.pop[top_idx] = cand
                            self.fitness[top_idx] = f_c

        return self.best_f, self.best_x