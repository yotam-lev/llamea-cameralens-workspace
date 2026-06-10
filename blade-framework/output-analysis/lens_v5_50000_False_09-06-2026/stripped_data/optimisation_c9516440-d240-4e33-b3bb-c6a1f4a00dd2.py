import numpy as n
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.pop_size = 35
        self.F_base = 0.9
        self.Cr = 0.8
        self.refine_freq = 10

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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        n_samples = self.pop_size
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        p_f = np.array([self._evaluate(pop[i]) for i in range(n_samples)])

        H_pd = None
        S_prev = None

        it = 0
        while self.evals < self.budget and it < 500:
            it += 1

            # Strict budget guard before Hessian query
            if self.evals >= self.budget:
                break
            if it % 12 == 0 or S_prev is None:
                idx_best = np.argmin(p_f)
                x_best = pop[idx_best]
                try:
                    H = hess_func(x_best)
                    eigs, V = np.linalg.eigh(H)
                    # Strict PD enforcement via eigenvalue absolute mapping
                    H_pd = V @ np.diag(np.abs(eigs)) @ V.T
                    # Preconditioner: inverse square root of curvature for mutation scaling
                    S_prev = V @ np.diag(1.0 / np.sqrt(np.abs(eigs) + 1e-8)) @ V.T
                except Exception:
                    pass

            # Hessian-Aware Differential Evolution
            for i in range(n_samples):
                if self.evals >= self.budget:
                    break
                r_idx = [j for j in range(n_samples) if j != i]
                r1, r2, r3 = np.random.choice(r_idx, 3, replace=False)

                diff = pop[r2] - pop[r3]
                if S_prev is not None:
                    diff = S_prev @ diff

                # Adaptive mutation factor based on spectral spread
                F = self.F_base * np.exp(-0.5 * np.log10(np.trace(S_prev @ S_prev.T) / 18)) if S_prev is not None else self.F_base

                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.Cr or j == j_rand:
                        if 18 <= j < 24:
                            # Categorical integer mutation
                            if np.random.rand() < 0.15:
                                trial[j] = np.random.randint(0, 6)
                        else:
                            trial[j] = pop[i, j] + F * diff[j]
                    trial = np.clip(trial, -1.0, 1.0)
                    trial[18:24] = np.clip(np.round(trial[18:24]), 0, 5).astype(int)

                f_trial = self._evaluate(trial)
                if f_trial <= p_f[i]:
                    p_f[i] = f_trial
                    pop[i] = trial.copy()
                    if f_trial < self.best_f:
                        self.best_f = f_trial
                        self.best_x = trial.copy()

            # Trust-Region Refinement on elites
            if self.evals < self.budget and it % self.refine_freq == 0 and S_prev is not None:
                top_idx = np.argsort(p_f)[:5]
                for idx in top_idx:
                    if self.evals >= self.budget:
                        break
                    cand = pop[idx].copy()
                    # Sufficiently large identity for regularization before solver
                    H_solver = H_pd + np.eye(18) * 1e-3
                    res = minimize(
                        lambda xc: self._evaluate(np.concatenate([xc, cand[18:24]])),
                        cand[:18], method='trust-constr',
                        hess=lambda xc: H_solver,
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 20, 'verbose': 0}
                    )
                    if res.success:
                        cand_loc = np.concatenate([res.x, cand[18:24]])
                        f_loc = self._evaluate(cand_loc)
                        if f_loc < p_f[idx]:
                            p_f[idx] = f_loc
                            pop[idx] = cand_loc.copy()
                            if f_loc < self.best_f:
                                self.best_f = f_loc
                                self.best_x = cand_loc.copy()

        return self.best_f, self.best_x