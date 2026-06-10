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
        self.w_min, self.w_max = 0.2, 0.8
        self.c1, self.c2 = 1.8, 1.8
        self.ravine_thresh = 55.0
        self.refine_freq_base = 18
        self.refine_freq_min = 5
        self.hess_func = None
        self.func = None
        self.H = None
        self.H_inv = None
        self.H_valid = False
        self.eig_vals = None
        self.eig_vecs = None
        self.improvement_hist = []
        self.cat_bias = np.ones(6) / 6.0
        self.stall_count = 0
        self.best_f_prev = float('inf')

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
            self.stall_count = 0
        else:
            self.stall_count += 1
        return f

    def _update_hessian(self, x):
        if self.evals >= self.budget:
            return
        try:
            H = self.hess_func(x)
            eigs, V = np.linalg.eigh(H)
            # Strict PD enforcement
            H_pd = V @ np.diag(np.abs(eigs)) @ V.T
            # Regularization for inversion stability
            H_reg = H_pd + 1e-3 * np.eye(18)
            self.H = H_pd
            self.H_inv = np.linalg.inv(H_reg)
            self.eig_vals = eigs
            self.eig_vecs = V
            self.H_valid = True
        except Exception:
            self.H_valid = False

    def _adaptive_params(self):
        budget_frac = self.evals / self.budget
        if len(self.improvement_hist) > 5:
            recent_improvement = self.best_f_prev - self.best_f
            self.improvement_hist.append(recent_improvement)
            avg_improvement = np.mean(self.improvement_hist[-10:])
            
            # Adaptive mutation scale: larger when stagnating, smaller when improving
            if recent_improvement < 1e-6 or recent_improvement < 0:
                self.stall_count += 1
            else:
                self.stall_count = max(0, self.stall_count - 1)
                
            stall_factor = np.clip(self.stall_count / 20.0, 0.0, 1.0)
            self.mut_scale = 0.2 + 1.5 * stall_factor + 0.5 * (1.0 - budget_frac)
            
            # Adaptive refinement frequency
            if recent_improvement < 1e-5:
                self.refine_freq = self.refine_freq_min
            elif recent_improvement > 0.01:
                self.refine_freq = int(np.clip(self.refine_freq_base * (1.0 + budget_frac), self.refine_freq_min, 30))
            else:
                self.refine_freq = self.refine_freq_base
                
            # Adaptive inertia
            self.w = self.w_max - (self.w_max - self.w_min) * budget_frac
        else:
            self.mut_scale = 0.5
            self.refine_freq = self.refine_freq_base
            self.w = 0.8

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.func = func
        self.hess_func = hess_func
        self.mut_scale = 0.5
        self.refine_freq = self.refine_freq_base
        self.w = 0.8

        n_samples = self.n_swarm
        # Initialize with categorical bias
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        # Biased categorical initialization
        for i in range(n_samples):
            for j in range(18, 24):
                cat_idx = np.random.choice(6, p=self.cat_bias)
                pop[i, j] = cat_idx + 0.5  # Center of category bin
        vel = np.random.uniform(-0.4, 0.4, size=(n_samples, self.dim))
        p_best = pop.copy()
        p_f = np.array([self._evaluate(pop[i]) for i in range(n_samples)])

        g_idx = np.argmin(p_f)
        g_best = pop[g_idx].copy()
        g_f = p_f[g_idx]

        it = 0
        while self.evals < self.budget and it < 400:
            it += 1
            self.best_f_prev = g_f
            self._adaptive_params()

            if self.evals < self.budget and (not self.H_valid or it % 12 == 0):
                self._update_hessian(g_best)

            cond = self.eig_vals.max() / max(self.eig_vals.min(), 1e-9) if self.H_valid else 0.0
            is_ravine = self.H_valid and cond > self.ravine_thresh

            for i in range(n_samples):
                r1, r2 = np.random.rand(2)
                v = vel[i].copy()

                if self.H_valid:
                    # Curvature-guided mutation
                    if is_ravine:
                        coords = self.eig_vecs.T @ v[:18]
                        for k in range(18):
                            lam = self.eig_vals[k]
                            if lam > 1.0: coords[k] *= 0.2
                            elif lam < 0.8: coords[k] *= 3.0
                        v[:18] = self.eig_vecs @ coords
                    else:
                        # Hessian-informed noise
                        noise = np.random.multivariate_normal(
                            0, self.H_inv * self.mut_scale
                        )
                        v[:18] += noise

                v[:18] = self.w * v[:18] + self.c1 * r1 * (p_best[i, :18] - pop[i, :18]) + self.c2 * r2 * (g_best[:18] - pop[i, :18])
                v[18:24] = self.w * v[18:24] + self.c2 * r1 * (p_best[i, 18:24] - pop[i, 18:24]) + self.c2 * r2 * (g_best[18:24] - pop[i, 18:24])

                pop[i] += v * 0.5

                # Biased categorical perturbation
                for j in range(18, 24):
                    if np.random.rand() < 0.1:
                        cat_idx = np.random.choice(6, p=self.cat_bias)
                        pop[i, j] = cat_idx + 0.5

                f = self._evaluate(pop[i])
                if f < p_f[i]:
                    p_f[i] = f
                    p_best[i] = pop[i].copy()
                    if f < g_f:
                        g_f = f
                        g_best = pop[i].copy()
                        g_idx = i
                        # Update categorical bias based on best solution
                        self.cat_bias = np.copy(self.cat_bias)
                        self.cat_bias[g_best[18:24]] += 1.0
                        self.cat_bias /= self.cat_bias.sum()

            vel = np.copy(vel)

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