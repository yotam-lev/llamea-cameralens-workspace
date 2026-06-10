import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_swarm = 40
        self.w = 0.65
        self.c1, self.c2 = 1.6, 1.6
        self.refine_freq = 15
        self.history = []
        self.improve_window = 20
        self.stall_threshold = 1e-6
        self.H_valid = False
        self.eig_vals = np.zeros(18)
        self.eig_vecs = np.eye(18)
        self.H = np.eye(18)

    def _evaluate(self, x):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = self.func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _adapt(self, g_f):
        self.history.append(g_f)
        if len(self.history) > self.improve_window:
            self.history.pop(0)
            
        if len(self.history) == self.improve_window:
            improvement = self.history[0] - self.history[-1]
            budget_progress = self.evals / self.budget
            
            # Stall detection and parameter tuning
            if improvement < self.stall_threshold:
                self.refine_freq = min(35, int(self.refine_freq * 1.1))
                self.w = max(0.25, self.w * 0.9)
            else:
                self.refine_freq = max(8, int(self.refine_freq * 0.9))
                self.w = min(0.85, self.w * 1.05)
                
            # Budget progress naturally pushes toward exploitation
            self.refine_freq = int(self.refine_freq * (1.0 + budget_progress * 0.5))
            
            # Hessian condition adaptation
            if self.H_valid:
                cond = self.eig_vals.max() / max(abs(self.eig_vals.min()), 1e-9)
                if cond > 60:
                    self.w = max(0.2, self.w * 0.7) # Dampen in stiff regions
                else:
                    self.w = min(0.85, self.w * 1.02)

    def _update_hessian(self, x):
        if self.evals >= self.budget:
            return
        try:
            H = self.hess_func(x)
            eigs, V = np.linalg.eigh(H)
            self.H = V @ np.diag(np.abs(eigs)) @ V.T
            self.eig_vals = eigs
            self.eig_vecs = V
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        pop = np.random.uniform(-1, 1, size=(self.n_swarm, self.dim))
        vel = np.random.uniform(-0.35, 0.35, size=(self.n_swarm, self.dim))
        p_best = pop.copy()
        p_f = np.array([self._evaluate(pop[i]) for i in range(self.n_swarm)])

        g_idx = np.argmin(p_f)
        g_best = pop[g_idx].copy()
        g_f = p_f[g_idx]

        it = 0
        while self.evals < self.budget and it < 400:
            it += 1
            
            self._adapt(g_f)
            
            if self.evals < self.budget and (not self.H_valid or it % 15 == 0):
                self._update_hessian(g_best)

            cond = self.eig_vals.max() / max(abs(self.eig_vals.min()), 1e-9) if self.H_valid else 0.0
            is_ravine = self.H_valid and cond > 55.0

            for i in range(self.n_swarm):
                r1, r2 = np.random.rand(2)
                v = vel[i].copy()

                if self.H_valid:
                    if is_ravine:
                        coords = self.eig_vecs.T @ v[:18]
                        for k in range(18):
                            lam = abs(self.eig_vals[k])
                            coords[k] *= (0.15 if lam > 1.0 else 3.5 if lam < 0.5 else 1.0)
                        v[:18] = self.eig_vecs @ coords
                    else:
                        v[:18] = self.H @ v[:18]

                vel[i, :18] = self.w * v[:18] + self.c1 * r1 * (p_best[i, :18] - pop[i, :18]) + self.c2 * r2 * (g_best[:18] - pop[i, :18])
                vel[i, 18:24] = self.w * v[18:24] + self.c2 * r1 * (p_best[i, 18:24] - pop[i, 18:24]) + self.c2 * r2 * (g_best[18:24] - pop[i, 18:24])
                
                pop[i] += vel[i] * 0.55
                
                f = self._evaluate(pop[i])
                if f < p_f[i]:
                    p_f[i] = f
                    p_best[i] = pop[i].copy()
                    if f < g_f:
                        g_f = f
                        g_best = pop[i].copy()
                        g_idx = i

            if self.evals < self.budget and it % self.refine_freq == 0 and not is_ravine:
                try:
                    res = minimize(
                        lambda xc: self._evaluate(np.concatenate([xc, g_best[18:24]])),
                        g_best[:18], method='trust-constr',
                        hess=lambda xc: self.H if self.H_valid else np.eye(18),
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 25, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, g_best[18:24]])
                        f_c = self._evaluate(cand)
                        if f_c < g_f:
                            g_f = f_c
                            g_best = cand.copy()
                except Exception:
                    pass

        return self.best_f, self.best_x