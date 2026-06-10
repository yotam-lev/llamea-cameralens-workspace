import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        self.pop_size = 16
        self.temp = 1.0
        self.temp_min = 0.05
        
    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        x = np.clip(x, -1.0, 1.0)
        x[18:24] = np.clip(np.round(x[18:24]), 0, 5).astype(int)
        f = func(x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f
        
    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_samples = self.pop_size
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        fitness = np.full(n_samples, float('inf'))
        
        for i in range(n_samples):
            if self.evals >= self.budget: break
            fitness[i] = self._evaluate(pop[i], func)
            
        best_idx = np.argmin(fitness)
        g_x = pop[best_idx].copy()
        g_f = fitness[best_idx]
        
        H_reg = None
        H_inv = None
        Q = None
        eigs = None
        it = 0
        
        while self.evals < self.budget:
            it += 1
            
            # PERIODIC HESSIAN UPDATE
            if hess_func is not None and it % 4 == 0:
                if self.evals < self.budget:
                    H = hess_func(g_x)
                    eigs, Q = np.linalg.eigh(H)
                    eigs = np.abs(eigs) + 1e-2
                    H_reg = Q @ np.diag(eigs) @ Q.T
                    H_inv = Q @ np.diag(1.0 / eigs) @ Q.T
                    
            if H_reg is None:
                H_reg = np.eye(18)
                H_inv = np.eye(18)
                eigs = np.ones(18)
                Q = np.eye(18)
                
            # --- DISCRETE -> CONTINUOUS COUPLING ---
            # Categorical materials dictate local curvature scaling
            cat_ids = np.clip(np.round(g_x[18:24]), 0, 5).astype(int)
            mat_scalers = np.array([0.4, 0.4, 1.0, 1.0, 2.5, 2.5])
            mod = np.repeat(mat_scalers[cat_ids], 3)
            joint_eigs = eigs * mod
            joint_inv = Q @ np.diag(1.0 / joint_eigs) @ Q.T
            
            # --- CONTINUOUS -> DISCRETE COUPLING ---
            # Continuous manifold flatness biases discrete search probabilities
            agg_curv = np.mean(joint_eigs)
            temp = max(self.temp_min, self.temp * 0.98)
            logits = -np.log(agg_curv + 1e-3) * np.ones(6)
            logits += np.random.gumbel(size=6) * temp
            p_discrete = np.exp(logits - np.max(logits))
            p_discrete /= np.sum(p_discrete)
            
            # Bidirectional update: discrete refinement
            if it % 2 == 0 and self.evals < self.budget:
                if np.random.random() < 0.35:
                    new_cat = np.random.choice(6, p=p_discrete)
                    cand_cat = cat_ids.copy()
                    cand_cat[0] = new_cat
                    cand_x = np.concatenate([g_x[:18], cand_cat]).astype(float)
                    cand_f = self._evaluate(cand_x, func)
                    if cand_f < g_f - 1e-4 or np.random.random() < np.exp(-(cand_f - g_f) / temp):
                        g_x[18:24] = cand_cat
                        g_f = cand_f
                        fitness[best_idx] = cand_f
                        
            # Bidirectional update: continuous trust-region
            if it % 2 == 0 and self.evals < self.budget:
                x_c = g_x[:18].copy()
                cat_ids = np.clip(np.round(g_x[18:24]), 0, 5).astype(int)
                
                def obj(xc):
                    return func(np.concatenate([xc, cat_ids]))
                    
                res = minimize(
                    obj, x_c, hess=lambda xc: H_reg,
                    method='trust-constr',
                    bounds=[(-1.0, 1.0)] * 18,
                    options={'maxiter': 20, 'verbose': 0}
                )
                cand_x = np.concatenate([res.x, cat_ids]).astype(float)
                cand_f = self._evaluate(cand_x, func)
                if cand_f < g_f:
                    g_f = cand_f
                    g_x = cand_x
                    fitness[best_idx] = cand_f
                    self.best_f = cand_f
                    self.best_x = cand_x.copy()
                    
            # POPULATION PERTURBATION
            if it % 5 == 0:
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    noise_c = np.random.multivariate_normal(np.zeros(18), joint_inv * 0.06)
                    noise_d = np.random.choice(6, size=6, p=p_discrete)
                    cand = np.concatenate([g_x[:18] + noise_c, g_x[18:24]])
                    cand = np.clip(cand, -1.0, 1.0)
                    cand[18:24] = np.clip(np.round(cand[18:24]), 0, 5).astype(int)
                    fitness[i] = self._evaluate(cand, func)
                best_idx = np.argmin(fitness)
                g_x = pop[best_idx].copy()
                g_f = fitness[best_idx]
                
        return self.best_f, self.best_x