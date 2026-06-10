import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_pop = 60
        self.history = []
        self.improve_window = 25
        self.stall_thresh = 1e-4
        self.H = np.eye(18)
        self.H_valid = False
        self.eig_vals = np.zeros(18)
        self.eig_vecs = np.eye(18)
        self.flat_mask = np.zeros(18, dtype=bool)
        self.stiff_mask = np.ones(18, dtype=bool)

    def _evaluate(self, x):
        if self.evals >= self.budget:
            return float('inf')
        # STRICT BOUNDARY & CATEGORICAL ENFORCEMENT
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        
        if self.evals >= self.budget:
            return float('inf')
        f = self.func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _update_hessian(self, x):
        if self.evals >= self.budget:
            break
        try:
            H_raw = self.hess_func(x)
            eigs, V = np.linalg.eigh(H_raw)
            # Regularization: absolute eigenvalues guarantee positive-definiteness
            self.H = V @ np.diag(np.abs(eigs)) @ V.T
            self.eig_vals = np.abs(eigs)
            self.eig_vecs = V
            self.H_valid = True
            # Subspace split based on curvature median
            median_curv = np.median(self.eig_vals)
            self.flat_mask = self.eig_vals < median_curv * 0.3
            self.stiff_mask = ~self.flat_mask
        except Exception:
            self.H_valid = False

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func

        pop = np.random.uniform(-1, 1, size=(self.n_pop, self.dim))
        f_vals = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            if self.evals >= self.budget: break
            f_vals[i] = self._evaluate(pop[i])

        idx_g = np.argmin(f_vals)
        g_best = pop[idx_g].copy()
        g_f = f_vals[idx_g]

        it = 0
        while self.evals < self.budget and it < 600:
            it += 1
            if self.evals >= self.budget: break
            
            self.history.append(g_f)
            if len(self.history) > self.improve_window:
                self.history.pop(0)

            stall = False
            if len(self.history) == self.improve_window:
                if self.history[0] - self.history[-1] < self.stall_thresh:
                    stall = True

            # Hessian update logic
            if self.evals < self.budget and (not self.H_valid or it % 12 == 0 or stall):
                self._update_hessian(g_best)

            for i in range(self.n_pop):
                if self.evals >= self.budget: break

                # NOVEL CURVATURE-CONDITIONAL DYNAMICS
                if stall and self.H_valid:
                    # ESCAPE PHASE: Levy flight along flat (low-curvature) subspace
                    step = np.random.randn(18)
                    step[self.stiff_mask] = 0.0
                    norm_f = np.linalg.norm(step)
                    if norm_f > 0: step /= norm_f
                    
                    # Scale inversely to flat curvature to ensure aggressive jumping
                    min_flat = max(self.eig_vals[self.flat_mask].min(), 1e-4)
                    levy_scale = 1.0 / np.sqrt(min_flat)
                    step *= np.random.standard_cauchy() * levy_scale * 0.6
                    pop[i, :18] += step

                    # Stochastic categorical perturbation
                    if np.random.rand() < 0.4:
                        c_idx = np.random.randint(18, 24)
                        pop[i, c_idx] = np.clip(pop[i, c_idx] + np.random.choice([-1, 1]), 0, 5)
                else:
                    # EXPLOITATION PHASE: Damped Newton step along stiff subspace
                    if self.H_valid:
                        dir_c = g_best[:18] - pop[i, :18]
                        if np.linalg.norm(dir_c) > 1e-6:
                            dir_c /= np.linalg.norm(dir_c)
                        # Regularized solve prevents degeneracy
                        H_reg = self.H + 1e-3 * np.eye(18)
                        try:
                            dx = np.linalg.solve(H_reg, dir_c) * 0.5
                            pop[i, :18] += dx
                        except Exception:
                            pass

                    # Gibbs-style categorical sampling
                    if np.random.rand() < 0.15:
                        for j in range(18, 24):
                            if np.random.rand() < 0.25:
                                pop[i, j] = np.random.randint(0, 6)

                # HARD ENFORCEMENT
                pop[i] = np.clip(pop[i], -1.0, 1.0)
                pop[i, 18:24] = np.clip(np.round(pop[i, 18:24]), 0, 5).astype(int)

                f_new = self._evaluate(pop[i])
                f_vals[i] = f_new
                if f_new < g_f:
                    g_f = f_new
                    g_best = pop[i].copy()

            # Periodic Trust-Region refinement on global best
            if self.evals < self.budget and it % 10 == 0 and self.H_valid:
                try:
                    res = minimize(
                        lambda xc: self._evaluate(np.concatenate([xc, g_best[18:24]])),
                        g_best[:18], method='trust-constr',
                        hess=lambda xc: self.H,
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