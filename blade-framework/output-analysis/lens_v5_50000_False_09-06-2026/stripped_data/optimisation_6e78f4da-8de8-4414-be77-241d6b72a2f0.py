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
        self.coupling_tau = 0.75
        self.hess_freq = 4
        self.last_hess_eval = -1
        self.H_reg = None
        self.H_inv_diag = None
        self.H_eigs = None
        self.H_Q = None
        
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

    def _update_hessian(self, x_full):
        H = hess_func(x_full)
        eigs, Q = np.linalg.eigh(H)
        # Regularize: ensure positive definiteness via absolute eigenvalues
        self.H_eigs = np.abs(eigs) + 1e-4
        self.H_Q = Q
        self.H_reg = Q @ np.diag(self.H_eigs) @ Q.T
        # Precompute inverse diagonal for curvature-aware weighting
        H_inv = Q @ np.diag(1.0 / self.H_eigs) @ Q.T
        self.H_inv_diag = np.diag(H_inv)
        
    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Strict LHS initialization syntax
        n_samples = self.pop_size
        self.pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        self.fitness = np.full(n_samples, float('inf'))
        
        # Initial evaluation
        for i in range(n_samples):
            if self.evals >= self.budget: break
            self.fitness[i] = self._evaluate(self.pop[i], func)
            
        iter_count = 0
        while self.evals < self.budget:
            iter_count += 1
            best_idx = np.argmin(self.fitness)
            g_x = self.pop[best_idx].copy()
            g_f = self.fitness[best_idx]
            
            # Hessian update for curvature estimation
            if hess_func is not None and (iter_count % self.hess_freq == 0 or self.H_reg is None):
                if self.evals < self.budget:
                    self._update_hessian(g_x)
                    self.last_hess_eval = self.evals
                    
            # CURVATURE-DRIVEN DISCONTINUOUS COUPLING
            if self.H_inv_diag is not None:
                # Aggregate continuous curvature to determine discrete mutation intensity
                agg_curv = np.mean(1.0 / (self.H_eigs + 1e-6))
                # Discrete selection probabilities weighted by curvature sensitivity
                logits = np.log(agg_curv) * np.ones(6)
                logits += np.random.gumbel(size=6) * 0.3
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                
                # Mutate categorical variables based on continuous landscape flatness
                for k in range(6):
                    if np.random.random() < self.coupling_tau * agg_curv:
                        g_x[18:24][k] = np.random.choice(6, p=probs)
                        
            # CONTINUOUS UPDATE: Hessian-preconditioned natural gradient
            if self.H_inv_diag is not None:
                if grad_func is not None:
                    g_full = grad_func(g_x)
                    g_c = g_full[:18]
                else:
                    g_c = np.zeros(18)
                    
                # Natural gradient direction (preconditioned by Hessian inverse)
                d_c = -np.diag(self.H_inv_diag) * (g_c + 1e-6)
                step_c = 0.1 * d_c
                g_x[:18] += step_c
                g_x[:18] = np.clip(g_x[:18], -1.0, 1.0)
                
            # MEMETIC REFINEMENT: Trust-region on continuous subspace
            if self.evals < self.budget and iter_count % 3 == 0:
                x_c = g_x[:18].copy()
                cat_ids = np.clip(np.round(g_x[18:24]), 0, 5).astype(int)
                
                def obj(xc):
                    return func(np.concatenate([xc, cat_ids]))
                    
                if hess_func is not None:
                    res = minimize(
                        obj, x_c, hess=lambda xc: self.H_reg,
                        method='trust-constr',
                        bounds=[(-1.0, 1.0)] * 18,
                        options={'maxiter': 20, 'verbose': 0}
                    )
                    if self.evals < self.budget:
                        cand = np.concatenate([res.x, cat_ids])
                        cand_f = self._evaluate(cand, func)
                        if cand_f < g_f:
                            g_f = cand_f
                            g_x = cand
                            self.fitness[best_idx] = cand_f
                            self.best_f = cand_f
                            self.best_x = cand.copy()
                            
            # POPULATION DIVERSITY: Curvature-aware perturbation
            if iter_count % 4 == 0:
                for i in range(n_samples):
                    if self.evals >= self.budget: break
                    noise = np.random.normal(0, 0.02, self.dim)
                    if self.H_inv_diag is not None:
                        # Anisotropic noise scaling based on inverse curvature
                        noise[:18] *= np.clip(np.sqrt(self.H_inv_diag), 0.05, 2.0)
                    self.pop[i] = g_x + noise
                    self.fitness[i] = self._evaluate(self.pop[i], func)
                g_x = self.pop[np.argmin(self.fitness)].copy()
                g_f = self.fitness[np.argmin(self.fitness)]
                
        return self.best_f, self.best_x