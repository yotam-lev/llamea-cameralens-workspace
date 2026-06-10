import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 45
        self.max_iters = 500
        self.temp = 1.0

    def _evaluate(self, x):
        if self.evals >= self.budget:
            break
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = self.func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        n_samples = 40
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        p_best = pop.copy()
        p_f = np.zeros(n_samples)

        for i in range(n_samples):
            if self.evals >= self.budget:
                break
            p_f[i] = self._evaluate(pop[i])

        g_idx = np.argmin(p_f)
        g_best = pop[g_idx].copy()
        g_f = p_f[g_idx]

        it = 0
        while self.evals < self.budget and it < self.max_iters:
            it += 1
            self.temp = max(0.05, self.temp * 0.99)

            if self.evals >= self.budget:
                break
                
            H_reg, V, eigs = None, None, None
            if it % 8 == 0:
                if self.evals >= self.budget:
                    break
                try:
                    H_raw = self.hess_func(g_best)
                    eigs, V = np.linalg.eigh(H_raw)
                    eigs_abs = np.abs(eigs)
                    H_reg = V @ np.diag(eigs_abs) @ V.T
                except Exception:
                    pass

            if H_reg is not None and V is not None:
                curv_inv = 0.4 / (eigs_abs + 1e-6)
                levy_dirs = np.random.standard_cauchy(size=(n_samples, 18))
                levy_scaled = levy_dirs * curv_inv[np.newaxis, :]
                levy_eig = levy_scaled @ V.T
            else:
                levy_eig = np.random.uniform(-0.5, 0.5, size=(n_samples, 18))

            for i in range(n_samples):
                if self.evals >= self.budget:
                    break
                    
                pop[i, :18] += levy_eig[i]
                
                cat_prob = np.exp(-self.temp)
                if np.random.rand() < cat_prob:
                    pop[i, 18:24] = np.random.randint(0, 6, size=6)
                    
                f = self._evaluate(pop[i])
                if f < p_f[i]:
                    p_f[i] = f
                    p_best[i] = pop[i].copy()
                    if f < g_f:
                        g_f = f
                        g_best = pop[i].copy()

            if it % 6 == 0 and self.evals < self.budget:
                elite_idx = np.argsort(p_f)[:3]
                for ei in elite_idx:
                    if self.evals >= self.budget:
                        break
                    try:
                        hess_fn = lambda xc: H_reg if H_reg is not None else np.eye(18)
                        res = minimize(
                            lambda xc: self._evaluate(np.concatenate([xc, g_best[18:24]])),
                            g_best[:18], method='trust-constr',
                            hess=hess_fn,
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 20, 'verbose': 0}
                        )
                        if res.success:
                            cand = np.concatenate([res.x, g_best[18:24]])
                            f_c = self._evaluate(cand)
                            if f_c < g_f:
                                g_f = f_c
                                g_best = cand.copy()
                            if f_c < p_f[ei]:
                                p_f[ei] = f_c
                                p_best[ei] = cand.copy()
                    except Exception:
                        pass

        return self.best_f, self.best_x