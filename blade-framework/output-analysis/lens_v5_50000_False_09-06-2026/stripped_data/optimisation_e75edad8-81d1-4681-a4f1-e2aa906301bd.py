import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

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

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        n_particles = 60
        pop_size = n_particles
        dim_c = 18
        dim_cat = 6
        
        # LHS Initialization
        pop = np.random.uniform(-1, 1, size=(n_particles, self.dim))
        pop_f = np.array([self._evaluate(x, func) for x in pop])
        
        pbest_pos = pop.copy()
        pbest_f = pop_f.copy()
        
        gbest_idx = np.argmin(pbest_f)
        gbest_pos = pop[gbest_idx].copy()
        gbest_f = pop_f[gbest_idx]
        
        vel = np.zeros_like(pop)
        
        # Hessian-Ellipsoid parameters
        H_PERIOD = 10
        LS_PERIOD = 5
        H_reg = np.eye(dim_c)
        ellipsoid_shape = np.eye(dim_c)
        
        w = 0.6
        c1 = 1.5
        c2 = 1.8
        
        while self.evals < self.budget:
            if self.evals >= self.budget: break
            
            # Update Hessian and Ellipsoid shape periodically
            if hess_func is not None and self.evals % H_PERIOD == 0:
                full_x = np.concatenate([gbest_pos[:dim_c], gbest_pos[dim_cat:]])
                if self.evals >= self.budget: break
                H_raw = hess_func(full_x)[:dim_c, :dim_c]
                eigs, Q = np.linalg.eigh(H_raw)
                
                # Regularization: ensure positive definite by taking abs eigenvalues
                H_reg = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                
                # Ellipsoid shape: axes proportional to inverse sqrt of curvature
                # This defines the search distribution for the swarm
                inv_sqrt_eigs = 1.0 / np.sqrt(np.abs(eigs) + 1e-8)
                ellipsoid_shape = Q @ np.diag(inv_sqrt_eigs) @ Q.T
                
            # Swarm Update
            for i in range(pop_size):
                if self.evals >= self.budget: break
                
                # Cognitive and Social components
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (pbest_pos[i] - pop[i])
                social = c2 * r2 * (gbest_pos - pop[i])
                
                # Hessian-warped stochastic term
                # Inject noise shaped by the ellipsoid to explore along low curvature
                stochastic = ellipsoid_shape @ np.random.randn(self.dim)
                stochastic[dim_c:] = 0.0  # Only shape continuous part
                
                # Velocity update
                vel[i] = w * vel[i] + cognitive + social + 0.1 * stochastic
                pop[i] += vel[i]
                
                # Categorical update: Swarm Consensus with mutation
                # If swarm is diverse, force categorical jumps
                diversity = np.mean(np.abs(pop - gbest_pos[:dim_c]))
                if diversity > 0.2:
                    p_mut = 0.3
                else:
                    p_mut = 0.05
                    
                if np.random.rand() < p_mut:
                    # Discrete mutation for cats
                    cats = pop[i, dim_c:]
                    mut = np.random.choice([-1, 0, 1], size=dim_cat)
                    cats_new = np.clip(np.round(cats + mut), 0, 5).astype(int)
                    pop[i, dim_c:] = cats_new
                else:
                    pop[i, dim_c:] = gbest_pos[dim_c:].copy()
            
            # Clip and evaluate
            pop = np.clip(pop, -1.0, 1.0)
            pop[18:24] = np.clip(np.round(pop[18:24]), 0, 5).astype(int)
            
            for i in range(pop_size):
                if self.evals >= self.budget: break
                f = self._evaluate(pop[i], func)
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_pos[i] = pop[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_idx = i
                    gbest_pos = pop[i].copy()
            
            # Local Trust-Region Refinement
            if self.evals % LS_PERIOD == 0 and hess_func is not None:
                xc = gbest_pos[:dim_c].copy()
                cat = gbest_pos[dim_c:].copy()
                full_x = np.concatenate([xc, cat])
                if self.evals >= self.budget: break
                H_raw = hess_func(full_x)[:dim_c, :dim_c]
                eigs, Q = np.linalg.eigh(H_raw)
                H_ls = Q @ np.diag(np.abs(eigs) + 1e-6) @ Q.T
                
                def obj(x_sub):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([x_sub, cat]))
                def jac(x_sub):
                    if self.evals >= self.budget: return np.zeros(dim_c)
                    if grad_func:
                        g = grad_func(np.concatenate([x_sub, cat]))
                        return g[:dim_c]
                    return np.zeros(dim_c)
                def hess(x_sub):
                    return H_ls

                res = minimize(obj, xc, jac=jac, hess=hess, method='trust-constr',
                               bounds=[(-1.0, 1.0)] * dim_c,
                               options={'maxiter': 15, 'verbose': 0})
                
                if self.evals < self.budget:
                    x_ref = np.concatenate([res.x, cat])
                    f_ref = self._evaluate(x_ref, func)
                    if f_ref < gbest_f:
                        gbest_f = f_ref
                        gbest_pos = x_ref.copy()
                    if f_ref < pbest_f[gbest_idx]:
                        pbest_f[gbest_idx] = f_ref
                        pbest_pos[gbest_idx] = x_ref.copy()
                    if f_ref < self.best_f:
                        self.best_f = f_ref
                        self.best_x = x_ref.copy()
                else:
                    break
        
        return self.best_f, self.best_x