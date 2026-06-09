import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # PSO Parameters
        self.n_particles = 25
        self.w = 0.7
        self.c1 = 2.0
        self.c2 = 2.0
        
        # State
        self.pop = np.zeros((self.n_particles, dim))
        self.vel = np.zeros((self.n_particles, dim))
        self.pbest = np.zeros((self.n_particles, dim))
        self.pbest_f = np.full(self.n_particles, float('inf'))
        self.gbest_idx = -1
        self.gbest_f = float('inf')
        self.gbest_x = np.zeros(dim)
        
        # Hessian cache
        self.H_scale = None
        self.last_hess_evals = 0
        
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

    def _regularize(self, H):
        eigs, Q = np.linalg.eigh(H)
        eps = 1e-6
        diag_vals = 1.0 / np.sqrt(np.abs(eigs) + eps)
        return Q @ np.diag(diag_vals) @ Q.T

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize Population
        self.pop = np.random.uniform(-1, 1, size=(self.n_particles, self.dim))
        self.vel = np.random.uniform(-1, 1, size=(self.n_particles, self.dim)) * 0.5
        
        for i in range(self.n_particles):
            if self.evals >= self.budget: break
            f = self._evaluate(self.pop[i], func)
            self.pbest[i] = self.pop[i].copy()
            self.pbest_f[i] = f
            if f < self.gbest_f:
                self.gbest_f = f
                self.gbest_idx = i
                self.gbest_x = self.pop[i].copy()
                self.best_f = f
                self.best_x = self.pop[i].copy()

        iter_count = 0
        while self.evals < self.budget:
            iter_count += 1
            
            # Update Hessian Scale (Preconditioner)
            if hess_func is not None and self.evals < self.budget:
                if iter_count % 5 == 0 or self.H_scale is None:
                    cat_ids = self.gbest_x[18:24].copy()
                    x_c = self.gbest_x[:18].copy()
                    full_x = np.concatenate([x_c, cat_ids])
                    H_raw = hess_func(full_x)
                    self.H_scale = self._regularize(H_raw)
                    self.last_hess_evals = self.evals

            # PSO Update
            r1 = np.random.random((self.n_particles, self.dim))
            r2 = np.random.random((self.n_particles, self.dim))
            
            vel_g = self.pop[self.gbest_idx]
            vel_p = self.pbest
            
            # Velocity update
            v_std = self.w * self.vel + self.c1 * r1 * (vel_p - self.pop) + self.c2 * r2 * (vel_g - self.pop)
            
            # Precondition via Hessian
            if self.H_scale is not None:
                for i in range(self.n_particles):
                    self.vel[i] = self.H_scale @ v_std[i]
            else:
                self.vel = v_std
                
            # Position update
            self.pop += self.vel
            
            # Evaluation and Update pbest/gbest
            for i in range(self.n_particles):
                if self.evals >= self.budget: break
                f = self._evaluate(self.pop[i], func)
                if f < self.pbest_f[i]:
                    self.pbest_f[i] = f
                    self.pbest[i] = self.pop[i].copy()
                if f < self.gbest_f:
                    self.gbest_f = f
                    self.gbest_idx = i
                    self.gbest_x = self.pop[i].copy()
                    self.best_f = f
                    self.best_x = self.pop[i].copy()
                    
            # Local Search on GBest
            if self.evals < self.budget and self.gbest_idx != -1:
                if hess_func is not None:
                    x_c = self.gbest_x[:18].copy()
                    cat_ids = self.gbest_x[18:24].copy()
                    full_x = np.concatenate([x_c, cat_ids])
                    H_reg = self._regularize(hess_func(full_x))
                    
                    def obj(xc): return func(np.concatenate([xc, cat_ids]))
                    def jac(xc): 
                        g = grad_func(np.concatenate([xc, cat_ids]))
                        return g[:18]
                    def hess(xc): return H_reg
                    
                    if self.evals < self.budget:
                        res = minimize(
                            obj, x_c, jac=jac, hess=hess,
                            method='trust-constr',
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 15}
                        )
                        if self.evals < self.budget:
                            cand = np.concatenate([res.x, cat_ids])
                            cand_f = self._evaluate(cand, func)
                            if cand_f < self.gbest_f:
                                self.gbest_f = cand_f
                                self.gbest_idx = self.gbest_idx
                                self.gbest_x = cand
                                self.best_f = cand_f
                                self.best_x = cand
                                self.pbest[self.gbest_idx] = cand
                                self.pbest_f[self.gbest_idx] = cand_f

        return self.best_f, self.best_x
