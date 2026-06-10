import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # Population & Covariance Parameters
        self.pop_size = 30
        self.sigma0 = 0.5
        self.cov = np.eye(self.dim)
        self.omega = 0.3  # Covariance decay factor
        
        # Hessian State
        self.H_reg = None
        self.H_Q = None
        self.H_eigs = None
        self.flat_idx = []
        self.neg_idx = []
        
        # Stagnation Tracking
        self.stag_cnt = 0
        self.last_best = float('inf')
        self.hess_freq = 4
        self.refine_freq = 5
        
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
        
    def _update_hessian(self, x):
        if hess_func is None:
            self.H_reg = np.eye(18)
            self.H_Q = np.eye(18)
            self.H_eigs = np.ones(18)
            return
        full_x = x.copy()
        full_x[18:24] = np.round(full_x[18:24]).astype(int)
        H = hess_func(full_x)
        H = 0.5 * (H + H.T)
        eigs, Q = np.linalg.eigh(H)
        eps = 1e-4
        self.H_eigs = np.abs(eigs) + eps
        self.H_Q = Q
        self.H_reg = Q @ np.diag(self.H_eigs) @ Q.T
        self.flat_idx = np.where(self.H_eigs < 1e-2)[0]
        self.neg_idx = np.where(eigs < 0)[0]
        
    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize population using strict LHS syntax
        pop = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        self._update_hessian(np.zeros(self.dim))
        
        while self.evals < self.budget:
            if self.evals >= self.budget: break
            
            # Evaluate population
            fits = np.array([self._evaluate(x, func) for x in pop])
            if self.evals >= self.budget: break
            
            best_idx = np.argmin(fits)
            
            # Stagnation detection
            if abs(self.last_best - self.best_f) < 1e-6:
                self.stag_cnt += 1
            else:
                self.stag_cnt = 0
            self.last_best = self.best_f
            
            # Periodic Hessian Update
            if self.evals % self.hess_freq == 0 or self.stag_cnt > 3:
                self._update_hessian(self.best_x)
                
            # Curvature-Deflected Perturbation Generation
            z = np.random.randn(self.pop_size, self.dim)
            # Inverse-sqrt preconditioning flattens anisotropic curvature
            inv_sqrt = 1.0 / np.sqrt(np.maximum(self.H_eigs, 1e-6))
            z_pre = z * inv_sqrt[np.newaxis, :]
            
            # Eigenmode Deflection: amplify traversal along flat/negative modes
            if len(self.neg_idx) > 0:
                neg_vecs = self.H_Q[:, self.neg_idx]
                defl_scale = 1.0 / np.abs(self.H_eigs[self.neg_idx])[np.newaxis, :]
                defl = np.random.randn(self.pop_size, len(self.neg_idx)) * defl_scale
                z_pre[:, self.neg_idx] += defl
                
            # Covariance-adapted step mapping
            z_deflected = z_pre @ np.sqrt(self.cov)
            pop = pop[best_idx][np.newaxis, :] + z_deflected * self.sigma0
            
            # Covariance MA-ELITE Update
            success_mask = fits < self.best_f
            if np.any(success_mask):
                success_pop = pop[success_mask] - pop[best_idx][np.newaxis, :]
                self.cov = self.omega * self.cov + (1 - self.omega) * np.cov(success_pop.T)
                if np.all(np.abs(self.cov) < 1e-12):
                    self.cov = np.eye(self.dim)
                    
            # Memetic Trust-Region Refinement on continuous subspace
            if self.evals < self.budget and self.evals % self.refine_freq == 0:
                xc = self.best_x[:18].copy()
                cat_ids = self.best_x[18:24].copy()
                
                def obj(xc): return func(np.concatenate([xc, cat_ids]))
                if grad_func is not None:
                    def jac(xc): return grad_func(np.concatenate([xc, cat_ids]))[:18]
                    res = minimize(obj, xc, jac=jac, hess=lambda x: self.H_reg, 
                                   method='trust-constr', bounds=[(-1.0, 1.0)]*18, 
                                   options={'maxiter': 15})
                else:
                    res = minimize(obj, xc, hess=lambda x: self.H_reg, 
                                   method='trust-constr', bounds=[(-1.0, 1.0)]*18, 
                                   options={'maxiter': 15})
                
                if res.success and self.evals < self.budget:
                    cand = np.concatenate([res.x, cat_ids])
                    cf = self._evaluate(cand, func)
                    if cf < self.best_f:
                        self.best_f = cf
                        self.best_x = cand
                        pop[best_idx] = cand
                        self.cov = np.eye(self.dim)  # Reset covariance after successful basin capture
                        self.stag_cnt = 0
                        
        return self.best_f, self.best_x