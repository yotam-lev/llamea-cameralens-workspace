import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.n_swarm = 48
        self.w = 0.6
        self.c1, self.c2 = 1.8, 1.8
        self.categorical_ids = np.arange(6)
        self.H = None
        self.H_pd = None
        self.H_valid = False
        self.regime_metric = np.eye(18)

    def _evaluate(self, x):
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

        n_samples = self.n_swarm
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        vel = np.random.uniform(-0.3, 0.3, size=(n_samples, self.dim))
        p_best = pop.copy()
        
        while self.evals < self.budget:
            # Hessian update & PD enforcement
            if self.evals < self.budget:
                if not self.H_valid or (self.evals % 25 == 0):
                    try:
                        H = self.hess_func(self.best_x)
                        eigs, V = np.linalg.eigh(H)
                        # Strict PD enforcement via absolute eigenvalues
                        self.H_pd = V @ np.diag(np.abs(eigs) + 1e-6) @ V.T
                        self.H_valid = True
                        self.regime_metric = self.H_pd * 0.15
                    except Exception:
                        pass

            coupling_freq = max(3, int(35 * (1 - self.evals / self.budget)))
            p_f = np.array([self._evaluate(pop[i]) for i in range(n_samples)])
            g_idx = np.argmin(p_f)
            g_best = pop[g_idx].copy()
            g_f = p_f[g_idx]

            it = 0
            while self.evals < self.budget and it < coupling_freq:
                it += 1
                for i in range(n_samples):
                    r1, r2 = np.random.rand(2)
                    
                    # Regime-guided preconditioning: discrete state dictates continuous metric
                    v_c = self.regime_metric @ vel[i, :18]
                    
                    # Continuous dynamics
                    v_c = self.w * v_c + self.c1 * r1 * (p_best[i, :18] - pop[i, :18]) + \
                           self.c2 * r2 * (g_best[:18] - pop[i, :18])
                    
                    # Categorical relaxation dynamics
                    v_cat = self.w * vel[i, 18:24] + self.c2 * r2 * (p_best[i, 18:24] - pop[i, 18:24]) + \
                            self.c2 * r2 * (g_best[18:24] - pop[i, 18:24])
                    
                    pop[i, :18] += v_c * 0.4
                    pop[i, 18:24] += v_cat * 0.2
                    
                    if self.evals < self.budget:
                        f = self._evaluate(pop[i])
                        if f < p_f[i]:
                            p_f[i] = f
                            p_best[i] = pop[i].copy()
                        if f < g_f:
                            g_f = f
                            g_best = pop[i].copy()
                
                vel = np.copy(vel)

            # Bidirectional Coupling & Refinement
            if self.evals < self.budget:
                # 1. Continuous -> Discrete: Greedy categorical sweep
                cat_ref = g_best.copy()
                for d in range(18, 24):
                    best_d_f = g_f
                    for val in self.categorical_ids:
                        cat_test = cat_ref.copy()
                        cat_test[d] = val
                        cat_test = np.clip(cat_test, -1.0, 1.0)
                        cat_test[18:24] = np.clip(np.round(cat_test[18:24]), 0, 5).astype(int)
                        if self.evals < self.budget:
                            f_test = self._evaluate(cat_test)
                            if f_test < best_d_f:
                                best_d_f = f_test
                                cat_ref[d] = val
                if best_d_f < g_f:
                    g_f = best_d_f
                    g_best = cat_ref.copy()

                # 2. Discrete -> Continuous: Conditional trust-region geometry optimization
                fixed_cat = g_best[18:24].copy()
                def cont_obj(xc):
                    xc = np.clip(xc, -1.0, 1.0)
                    return self._evaluate(np.concatenate([xc, fixed_cat]))

                def cont_hess(xc):
                    return self.H_pd if self.H_valid else np.eye(18)

                if self.H_valid:
                    try:
                        res = minimize(cont_obj, g_best[:18], method='trust-constr',
                                     hess=cont_hess, bounds=[(-1.0, 1.0)]*18,
                                     options={'maxiter': 35, 'verbose': 0})
                        if res.fun < g_f:
                            g_f = res.fun
                            g_best = np.concatenate([res.x, fixed_cat])
                            g_best[18:24] = fixed_cat
                    except Exception:
                        pass

        return self.best_f, self.best_x