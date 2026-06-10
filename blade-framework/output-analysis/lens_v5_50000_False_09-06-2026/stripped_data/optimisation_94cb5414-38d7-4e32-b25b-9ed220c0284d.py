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
        self.F = 0.5
        self.CR = 0.8
        self.n_pop = 20
        self.ls_freq = 10
        self.hess_freq = 15
        
        # History & Hessian state
        self.pop = None
        self.fitness = None
        self.best_f_hist = []
        self.improv_hist = []
        self.cond_hist = []
        self.stagnation = 0
        self.H_reg = None
        self.H_inv_diag = None

    def _clip_and_eval(self, x, func):
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
        if self.evals >= self.budget:
            return
        try:
            H = self._hess_func(x)
            eigs = np.linalg.eigvalsh(H)
            shift = max(0, 1e-3 - eigs.min())
            self.H_reg = H + shift * np.eye(18)
            self.H_inv_diag = 1.0 / np.clip(np.abs(eigs), 1e-4, None)
            self.cond = np.max(np.abs(eigs)) / max(np.min(np.abs(eigs)), 1e-6)
            self.cond_hist.append(self.cond)
            if len(self.cond_hist) > 15:
                self.cond_hist.pop(0)
        except Exception:
            if self.H_reg is None:
                self.H_reg = np.eye(18) * 1e-2
                self.H_inv_diag = np.ones(18) / 1e-2
                self.cond = 1.0

    def _adapt(self):
        self.best_f_hist.append(self.best_f)
        if len(self.best_f_hist) > 20:
            self.best_f_hist.pop(0)
            
        if len(self.best_f_hist) > 1:
            recent = self.best_f_hist[-2] - self.best_f_hist[-1]
            self.improv_hist.append(recent)
            if len(self.improv_hist) > 15:
                self.improv_hist.pop(0)
                
            mean_imp = np.mean(self.improv_hist[-10:]) if self.improv_hist else 0.0
            mean_cond = np.mean(self.cond_hist[-10:]) if self.cond_hist else 1.0
            progress = self.evals / self.budget
            
            if recent < 1e-8:
                self.stagnation += 1
            else:
                self.stagnation = max(0, self.stagnation - 1)
                
            # Adaptive Scaling Rules
            if self.stagnation > 4:
                self.F = min(1.2, self.F * 1.05)
                self.ls_freq = max(3, self.ls_freq // 2)
            else:
                self.F = max(0.2, self.F * 0.99)
                
            if mean_cond > 100 or self.stagnation > 6:
                self.n_pop = min(60, self.n_pop + 1)
            elif mean_imp > 1e-3 and self.n_pop > 15:
                self.n_pop = max(10, self.n_pop - 1)
                
            if progress > 0.8:
                self.ls_freq = max(10, self.ls_freq + 5)
            elif mean_imp < 1e-5:
                self.ls_freq = max(3, self.ls_freq - 1)
                
            # Dynamic Population Resizing
            if self.pop is not None:
                curr_n = len(self.fitness)
                if self.n_pop > curr_n:
                    needed = self.n_pop - curr_n
                    pop = np.random.uniform(-1, 1, size=(needed, self.dim))
                    self.pop = np.vstack([self.pop, pop])
                    self.fitness = np.concatenate([self.fitness, np.full(needed, np.inf)])
                elif self.n_pop < curr_n:
                    idx = np.argsort(self.fitness)[:self.n_pop]
                    self.pop = self.pop[idx]
                    self.fitness = self.fitness[idx]

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        self.best_f_hist = []
        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        self.pop = pop
        self.fitness = np.full(self.n_pop, np.inf)
        
        # Initial Evaluation
        for i in range(len(self.fitness)):
            if self.evals >= self.budget: break
            self.fitness[i] = self._clip_and_eval(self.pop[i], func)
            
        it = 0
        while self.evals < self.budget:
            it += 1
            self._adapt()
            
            # Hessian Update
            if self._hess_func and (it % self.hess_freq == 0 or self.H_reg is None):
                best_idx = np.argmin(self.fitness)
                self._update_hessian(self.pop[best_idx])
                
            best_idx = np.argmin(self.fitness)
            g_x = self.pop[best_idx].copy()
            
            # Differential Evolution Loop
            for i in range(len(self.fitness)):
                if self.evals >= self.budget: break
                
                idx = np.random.choice(len(self.fitness), 3, replace=False)
                a, b, c = self.pop[idx[0]], self.pop[idx[1]], self.pop[idx[2]]
                diff = a[:18] - b[:18]
                
                # Curvature-Aligned Mutation
                if self.H_inv_diag is not None:
                    diff *= np.sqrt(self.H_inv_diag[:18])
                    
                mutant = g_x.copy()
                mutant[:18] += self.F * np.concatenate([diff, np.zeros(6)])
                
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
                f_m = self._clip_and_eval(mutant, func)
                
                if f_m < self.fitness[i]:
                    self.pop[i] = mutant
                    self.fitness[i] = f_m
                    
            # Adaptive Trust-Region Local Search
            if self.evals < self.budget and it % self.ls_freq == 0:
                top_idx = np.argmin(self.fitness)
                c_x = self.pop[top_idx][:18].copy()
                cat = np.clip(np.round(self.pop[top_idx][18:24]), 0, 5).astype(int)
                
                if self.H_reg is not None:
                    try:
                        res = minimize(
                            lambda xc: func(np.concatenate([xc, cat])),
                            c_x, method='trust-constr',
                            hess=lambda xc: self.H_reg,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 25, 'verbose': 0}
                        )
                        if res.success and self.evals < self.budget:
                            cand = np.concatenate([res.x, cat])
                            f_c = self._clip_and_eval(cand, func)
                            if f_c < self.fitness[top_idx]:
                                self.pop[top_idx] = cand
                                self.fitness[top_idx] = f_c
                    except Exception:
                        pass
                        
        return self.best_f, self.best_x