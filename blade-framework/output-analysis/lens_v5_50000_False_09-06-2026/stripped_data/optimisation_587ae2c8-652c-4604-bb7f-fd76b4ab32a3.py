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
        self.n_particles = 30
        self.w = 0.6
        self.c1 = 1.8
        self.c2 = 1.8
        self.levy_alpha = 1.5
        
        # Escape Parameters
        self.levy_freq = 8
        self.stagnation_threshold = 0.5
        self.last_gbest_f = float('inf')
        self.stagnation_count = 0
        
        # State
        self.pop = np.zeros((self.n_particles, dim))
        self.vel = np.zeros((self.n_particles, dim))
        self.pbest = np.zeros((self.n_particles, dim))
        self.pbest_f = np.full(self.n_particles, float('inf'))
        self.gbest_idx = -1
        self.gbest_f = float('inf')
        self.gbest_x = np.zeros(dim)
        
        # Hessian state
        self.H_reg = None
        self.H_Q = None
        self.H_eigs = None
        self.last_hess_eval = -1
        
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

    def _get_hessian_data(self, x_c, cat_ids):
        full_x = np.concatenate([x_c, cat_ids])
        H = hess_func(full_x)
        eigs, Q = np.linalg.eigh(H)
        eps = 1e-4
        # Regularize for solver: ensure positive definite
        self.H_eigs = np.abs(eigs) + eps
        self.H_reg = Q @ np.diag(self.H_eigs) @ Q.T
        self.H_Q = Q
        return self.H_reg

    def _levy_anisotropic_step(self, n_steps, scale=0.05):
        n = self.dim
        L = np.random.standard_cauchy((n_steps, n))
        L = np.sign(L) * np.abs(L) ** (1.0 / self.levy_alpha)
        
        if self.H_Q is not None and self.H_eigs is not None:
            # Scale by inverse sqrt of eigenvalues to flatten curvature
            # Use safe scaling: 1/sqrt(max(eig, eps))
            scaling = 1.0 / np.sqrt(np.maximum(self.H_eigs, 1e-6))
            # Transform noise to anisotropic space
            L_scaled = self.H_Q @ (np.diag(scaling) @ (self.H_Q.T @ L.T)).T
            return L_scaled * scale
        else:
            return L * scale

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # Initialize Population
        self.pop = np.random.uniform(-1, 1, size=(self.n_particles, self.dim))
        self.vel = np.random.uniform(-1, 1, size=(self.n_particles, self.dim)) * 0.2
        
        # Initial Evaluation
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
                self.last_gbest_f = f

        iter_count = 0
        while self.evals < self.budget:
            iter_count += 1
            
            # Check for stagnation to trigger escape
            escape_trigger = False
            if self.gbest_idx != -1:
                improvement = abs(self.last_gbest_f - self.gbest_f)
                if improvement < self.stagnation_threshold:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0
                    self.last_gbest_f = self.gbest_f
                    
                if self.stagnation_count > 5 or (iter_count % self.levy_freq == 0 and hess_func is not None):
                    escape_trigger = True

            # Hessian Update for Anisotropy
            if hess_func is not None and (escape_trigger or iter_count % 10 == 0 or self.H_Q is None):
                if self.evals < self.budget:
                    cat_ids = self.gbest_x[18:24].copy()
                    x_c = self.gbest_x[:18].copy()
                    self._get_hessian_data(x_c, cat_ids)
                    self.last_hess_eval = self.evals

            # Levy Flight Escape
            if escape_trigger and self.H_Q is not None:
                L_steps = self._levy_anisotropic_step(self.n_particles, scale=0.15)
                # Apply to population and velocity to kick out of basins
                self.pop += L_steps
                self.vel += 0.5 * L_steps
                # Reset stagnation
                self.stagnation_count = 0

            # PSO Update
            r1 = np.random.random((self.n_particles, self.dim))
            r2 = np.random.random((self.n_particles, self.dim))
            
            vel_g = self.pop[self.gbest_idx]
            vel_p = self.pbest
            
            # Velocity update
            v_std = self.w * self.vel + self.c1 * r1 * (vel_p - self.pop) + self.c2 * r2 * (vel_g - self.pop)
            
            # Apply Hessian preconditioning to velocity for direction alignment
            if self.H_Q is not None:
                # Precondition: v_pre = Q diag(1/sqrt(eig)) Q^T v
                # This aligns steps with natural gradient directions
                temp = self.H_Q.T @ v_std.T
                temp = temp / np.sqrt(np.maximum(self.H_eigs, 1e-6))
                v_pre = self.H_Q @ temp
                v_pre = v_pre.T
                self.vel = v_pre
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
                    self.last_gbest_f = f

            # Local Search on GBest (Memetic Step)
            if self.evals < self.budget and self.gbest_idx != -1:
                if iter_count % 3 == 0 and hess_func is not None:
                    x_c = self.gbest_x[:18].copy()
                    cat_ids = self.gbest_x[18:24].copy()
                    
                    def obj(xc): return func(np.concatenate([xc, cat_ids]))
                    
                    if grad_func is not None:
                        def jac(xc): 
                            g = grad_func(np.concatenate([xc, cat_ids]))
                            return g[:18]
                        res = minimize(
                            obj, x_c, jac=jac, hess=lambda x: self.H_reg,
                            method='trust-constr',
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 10}
                        )
                    else:
                        res = minimize(
                            obj, x_c, hess=lambda x: self.H_reg,
                            method='trust-constr',
                            bounds=[(-1.0, 1.0)] * 18,
                            options={'maxiter': 10}
                        )
                        
                    if res.success and self.evals < self.budget:
                        cand = np.concatenate([res.x, cat_ids])
                        cand_f = self._evaluate(cand, func)
                        if cand_f < self.gbest_f:
                            self.gbest_f = cand_f
                            self.gbest_x = cand
                            self.best_f = cand_f
                            self.best_x = cand
                            self.pbest_f[self.gbest_idx] = cand_f
                            self.pbest[self.gbest_idx] = cand
                            self.last_gbest_f = cand_f

        return self.best_f, self.best_x