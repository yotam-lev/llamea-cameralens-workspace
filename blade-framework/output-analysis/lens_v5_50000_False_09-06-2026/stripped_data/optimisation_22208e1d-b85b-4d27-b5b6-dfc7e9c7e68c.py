import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 32
        self.C = np.eye(18)
        self.disc_probs = np.ones(6) / 6.0
        self.curr_regime = np.zeros(6, dtype=int)
        self.manifold_V = np.eye(18)
        self.stall_win = 20
        self.f_hist = []
        self.regime_locked = False

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

    def _update_manifold_coupling(self, x_cand):
        if self.evals >= self.budget:
            return
        
        disc_cand = x_cand[18:24].astype(int)
        regime_changed = not np.array_equal(disc_cand, self.curr_regime)
        
        if regime_changed and not self.regime_locked:
            self.regime_locked = True
            self.curr_regime = disc_cand.copy()
            
            if self.hess_func:
                try:
                    H = self.hess_func(x_cand)
                    eigs, V = np.linalg.eigh(H)
                    H_pd = V @ np.diag(np.abs(eigs)) @ V.T
                    self.manifold_V = V
                    self.C = H_pd + 1e-2 * np.eye(18)
                except Exception:
                    pass
            return

        if not regime_changed:
            sens = np.zeros(6)
            f_base = self.best_f
            for d in range(6):
                base_val = disc_cand[d]
                best_delta = 0.0
                for v in range(6):
                    if v == base_val:
                        continue
                    x_sens = x_cand.copy()
                    x_sens[18:d] = v
                    f_s = self._evaluate(x_sens)
                    delta = f_base - f_s
                    if delta > best_delta:
                        best_delta = delta
                sens[d] = max(0.0, best_delta)
            
            self.disc_probs = np.exp(sens) + 1e-4
            self.disc_probs /= self.disc_probs.sum()

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        for i in range(self.n_pop):
            if self.evals >= self.budget:
                break
            self._evaluate(pop[i])

        if self.evals >= self.budget:
            return self.best_f, self.best_x

        it = 0
        while self.evals < self.budget and it < 400:
            it += 1
            
            if self.evals < self.budget and (not self.regime_locked or it % 4 == 0):
                for d in range(6):
                    idx = np.random.choice(6, p=self.disc_probs)
                    pop[:, 18:d] = idx
                self.regime_locked = False

            for i in range(self.n_pop):
                if self.evals >= self.budget:
                    break
                step = np.random.randn(18) @ self.C
                pop[i, :18] += step * 0.12
                pop[i, 18:24] += 0.35 * (np.random.randn(6) - 0.5)
                
                f = self._evaluate(pop[i])
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = pop[i].copy()
                    if self.evals < self.budget:
                        self._update_manifold_coupling(pop[i])

            if self.evals < self.budget and it % 15 == 0:
                self.f_hist.append(self.best_f)
                if len(self.f_hist) > self.stall_win:
                    self.f_hist.pop(0)
                imp = self.f_hist[0] - self.f_hist[-1]
                
                if imp < 5e-5 and self.evals > self.budget * 0.1:
                    disc_fixed = self.best_x[18:24].copy()
                    local_improved = False
                    for d in range(6):
                        for v in range(6):
                            if self.evals >= self.budget: break
                            x_t = self.best_x.copy()
                            x_t[18:d] = v
                            f_t = self._evaluate(x_t)
                            if f_t < self.best_f:
                                self.best_f = f_t
                                self.best_x = x_t.copy()
                                local_improved = True
                                self.curr_regime = x_t[18:24].astype(int)
                        if local_improved: break
                    if local_improved:
                        self.C = np.eye(18)

            if self.evals < self.budget and it % 22 == 0:
                disc_fixed = self.best_x[18:24].copy()
                try:
                    res = minimize(
                        lambda xc: self._evaluate(np.concatenate([xc, disc_fixed])),
                        self.best_x[:18], method='trust-constr',
                        hess=lambda xc: self.C if np.linalg.cond(self.C) < 100 else np.eye(18),
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 30, 'verbose': 0}
                    )
                    if res.success:
                        cand = np.concatenate([res.x, disc_fixed])
                        f_c = self._evaluate(cand)
                        if f_c < self.best_f:
                            self.best_f = f_c
                            self.best_x = cand.copy()
                            if self.evals < self.budget:
                                self._update_manifold_coupling(cand)
                except Exception:
                    pass

        return self.best_f, self.best_x