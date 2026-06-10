import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # Swarm parameters
        self.n_particles = 25
        self.w = 0.6
        self.c1 = 1.8
        self.c2 = 1.8
        
        self.swarm = None
        self.vel = None
        self.pbest = None
        self.pbest_f = None
        self.gbest = None
        self.gbest_f = float('inf')
        
        self.hess_func = None
        self.H_reg = None
        self.D = None
        
        # Control parameters
        self.hess_freq = 8
        self.ls_freq = 5
        self.iter = 0

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
        if self.hess_func is None:
            return
        H = self.hess_func(x)
        eigs = np.linalg.eigvalsh(H)
        shift = max(0, 1e-6 - eigs.min())
        H_p = H + shift * np.eye(18)
        eigs_p = np.linalg.eigvalsh(H_p)
        eigs_p = np.maximum(eigs_p, 1e-6)
        self.H_reg = H_p
        self.D = 1.0 / np.sqrt(eigs_p)

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self.hess_func = hess_func
        self.func_ref = func
        
        # Initialize swarm
        self.swarm = np.random.uniform(-1, 1, size=(self.n_particles, self.dim))
        self.vel = np.zeros_like(self.swarm)
        self.pbest = self.swarm.copy()
        self.pbest_f = np.full(self.n_particles, np.inf)
        self.gbest = None
        self.gbest_f = float('inf')
        self.iter = 0
        
        while self.evals < self.budget:
            self.iter += 1
            
            # Update Hessian periodically on global best
            if self.iter % self.hess_freq == 0:
                if self.gbest is not None:
                    self._update_hessian(self.gbest)
            
            # Swarm update
            for i in range(self.n_particles):
                if self.evals >= self.budget: break
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Preconditioned velocity update for continuous part
                if self.D is not None:
                    vel_c = self.vel[i, :18] * self.D[:18]
                else:
                    vel_c = self.vel[i, :18]
                
                vel_c = self.w * vel_c + self.c1 * r1[:18] * (self.pbest[i, :18] - self.swarm[i, :18]) + self.c2 * r2[:18] * (self.gbest[:18] - self.swarm[i, :18])
                
                self.vel[i, :18] = vel_c
                self.swarm[i, :18] += vel_c
                
                # Standard velocity update for categorical part
                self.vel[i, 18:] = self.w * self.vel[i, 18:] + self.c1 * r1[18:] * (self.pbest[i, 18:] - self.swarm[i, 18:]) + self.c2 * r2[18:] * (self.gbest[18:] - self.swarm[i, 18:])
                self.swarm[i, 18:] += self.vel[i, 18:]
                
                # Boundary and discretization enforcement
                self.swarm[i, :18] = np.clip(self.swarm[i, :18], -1.0, 1.0)
                self.swarm[i, 18:24] = np.clip(np.round(self.swarm[i, 18:24]), 0, 5).astype(int)
                
                # Evaluate
                f = self._evaluate(self.swarm[i], func)
                
                # Update personal best
                if f < self.pbest_f[i]:
                    self.pbest[i] = self.swarm[i].copy()
                    self.pbest_f[i] = f
                
                # Update global best
                if f < self.gbest_f:
                    self.gbest = self.swarm[i].copy()
                    self.gbest_f = f
                    self.best_f = f
                    self.best_x = self.swarm[i].copy()
            
            # Local search on global best using trust-constr
            if self.evals < self.budget and self.iter % self.ls_freq == 0 and self.gbest is not None and self.H_reg is not None:
                c = self.gbest[:18].copy()
                cat = self.gbest[18:24].copy()
                
                res = minimize(
                    lambda xc: func(np.concatenate([xc, cat])),
                    c, method='trust-constr', hess=lambda xc: self.H_reg,
                    bounds=[(-1.0, 1.0)]*18, options={'maxiter': 50, 'verbose': 0}
                )
                
                if res.success and self.evals < self.budget:
                    cand = np.concatenate([res.x, cat])
                    f_c = self._evaluate(cand, func)
                    if f_c < self.gbest_f:
                        self.gbest = cand
                        self.gbest_f = f_c
                        self.best_f = f_c
                        self.best_x = cand.copy()
                        
        return self.best_f, self.best_x