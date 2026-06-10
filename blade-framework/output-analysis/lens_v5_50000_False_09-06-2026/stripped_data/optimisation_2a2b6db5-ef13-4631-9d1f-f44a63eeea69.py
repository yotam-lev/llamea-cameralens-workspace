import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        
        # SA Parameters
        self.T = 0.5
        self.T_min = 1e-7
        self.cooling_rate = 0.991
        self.init_pop_size = 30
        
        # Hessian State
        self.H_reg = None
        self.s_inv_sqrt = np.ones(18)
        self.hess_update_freq = 45
        self.ls_update_freq = 35
        
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
        
    def _update_hessian(self, x_c, cat_ids, hess_func):
        if self.evals >= self.budget:
            return
        full_x = np.concatenate([x_c, cat_ids])
        H = hess_func(full_x)
        eigs, Q = np.linalg.eigh(H)
        eps = 1e-4
        # Curvature-aware step sizing & PD regularization
        self.s_inv_sqrt = 1.0 / np.sqrt(np.abs(eigs) + eps)
        self.H_reg = Q @ np.diag(np.abs(eigs) + eps) @ Q.T
        
    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # LHS-compliant initialization
        pop = np.random.uniform(-1, 1, size=(self.init_pop_size, self.dim))
        self.x_curr = pop[0]
        if self.evals >= self.budget: return self.best_f, self.best_x
        self.f_curr = self._evaluate(self.x_curr, func)
        
        it = 0
        while self.T > self.T_min and self.evals < self.budget:
            it += 1
            
            # Hessian Update for Curvature-Adaptive Steps
            if hess_func is not None and (it == 1 or it % self.hess_update_freq == 0):
                self._update_hessian(self.best_x[:18], self.best_x[18:24], hess_func)
                
            # Continuous Proposal: Scale noise by inverse sqrt curvature
            step_cont = self.s_inv_sqrt * self.init_pop_size * self.T
            delta_cont = np.random.randn(18) * step_cont
            x_prop_cont = self.x_curr[:18] + delta_cont
            
            # Categorical Proposal: Discrete jumps in material space
            delta_cat = np.random.choice([-1, 0, 1], size=6)
            x_prop_cat = self.x_curr[18:24] + delta_cat
            
            x_prop = np.concatenate([x_prop_cont, x_prop_cat])
            
            if self.evals >= self.budget: break
            f_prop = self._evaluate(x_prop, func)
            
            # Metropolis Acceptance
            if f_prop < self.f_curr:
                accept = True
            else:
                df = f_prop - self.f_curr
                accept = np.random.rand() < np.clip(np.exp(-df / max(self.T, 1e-12)), 0, 1)
                
            if accept:
                self.x_curr = x_prop
                self.f_curr = f_prop
                
            # Adaptive Cooling Schedule
            if it % 60 == 0:
                self.T *= self.cooling_rate
                
            # Periodic Trust-Region Refinement on Global Best
            if self.evals < self.budget and it % self.ls_update_freq == 0 and hess_func is not None and self.H_reg is not None:
                x_c_ls = self.best_x[:18].copy()
                cat_ls = self.best_x[18:24].copy()
                
                def obj_ls(xc):
                    if self.evals >= self.budget: return float('inf')
                    return func(np.concatenate([xc, cat_ls]))
                    
                if grad_func is not None:
                    def jac_ls(xc):
                        if self.evals >= self.budget: return np.zeros(18)
                        return grad_func(np.concatenate([xc, cat_ls]))[:18]
                    res = minimize(obj_ls, x_c_ls, jac=jac_ls, hess=lambda x: self.H_reg,
                                   method='trust-constr', bounds=[(-1.0, 1.0)]*18,
                                   options={'maxiter': 12, 'verbose': 0})
                else:
                    res = minimize(obj_ls, x_c_ls, hess=lambda x: self.H_reg,
                                   method='trust-constr', bounds=[(-1.0, 1.0)]*18,
                                   options={'maxiter': 12, 'verbose': 0})
                    
                if res.success and self.evals < self.budget:
                    cand = np.concatenate([res.x, cat_ls])
                    cand_f = self._evaluate(cand, func)
                    if cand_f < self.best_f:
                        self.best_f = cand_f
                        self.best_x = cand
                        self.f_curr = self._evaluate(cand, func)
                        self.x_curr = cand.copy()
                        
        return self.best_f, self.best_x