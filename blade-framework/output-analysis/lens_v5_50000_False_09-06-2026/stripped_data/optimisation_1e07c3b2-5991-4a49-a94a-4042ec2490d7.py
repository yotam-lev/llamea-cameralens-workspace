import numpy as np
from scipy.optimize import minimize
import cma

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self._hess_func = None
        self.H_precond = None
        self.stagnation_counter = 0
        self.prev_best = float('inf')
        
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
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        return f

    def _compute_preconditioner(self, x):
        if self._hess_func is None:
            return None
        H = self._hess_func(x)
        eigs, Q = np.linalg.eigh(H)
        # Regularize to ensure positive definiteness
        reg_eigs = np.abs(eigs) + 1e-6
        self.H_precond = Q @ np.diag(1.0 / reg_eigs) @ Q.T
        return self.H_precond

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        self._hess_func = hess_func
        
        # LHS Initialization
        n_samples = 16
        pop = np.random.uniform(-1, 1, size=(n_samples, self.dim))
        fvals_init = np.array([self._evaluate(x, func) for x in pop])
        
        # Adaptive CMA-ES with Hessian Preconditioning
        if self.evals < self.budget:
            x0 = self.best_x[:18].copy()
            sigma0 = 0.2
            C_precond = self.H_precond if self.H_precond is not None else None
            
            # Cap CMA iterations to respect remaining budget
            cma_maxfevals = max(10, int((self.budget - self.evals) * 0.45))
            
            def cma_obj(xc):
                xc = np.clip(xc, -1.0, 1.0)
                x_full = np.concatenate([xc, self.best_x[18:24]])
                return self._evaluate(x_full, func)

            try:
                cms = cma.CMAEvolutionStrategy(x0, sigma0, 
                                                inopts={'bounds': [-1.0, 1.0], 
                                                        'CovMatrix': C_precond,
                                                        'popsize': 16,
                                                        'maxfevals': cma_maxfevals})
                while not cms.stop():
                    solutions = cms.ask()
                    fvals_c = [cma_obj(xc) for xc in solutions]
                    if self.evals >= self.budget: break
                    cms.tell(solutions, fvals_c)
                best_c = cms.result.xbest
                best_f_c = cms.result.fbest
            except Exception:
                best_c = x0
                best_f_c = float('inf')
                
            if best_f_c < self.best_f:
                self.best_f = best_f_f
                self.best_x = np.concatenate([best_c, self.best_x[18:24]])
                
            # Curvature-Guided Discrete Resonance
            # Swap IDs corresponding to high preconditioner diagonal elements (low curvature)
            if self.H_precond is not None:
                diag_var = np.diag(self.H_precond)
                # Identify directions where continuous geometry is highly uncertain/shallow
                threshold = np.quantile(diag_var[:18], 0.8)
                swap_mask = diag_var[:18] > threshold
                
                if np.any(swap_mask):
                    perturb = np.random.randint(-1, 2, 6)
                    cat_ids = np.clip(np.round(self.best_x[18:24]) + perturb, 0, 5).astype(int)
                else:
                    cat_ids = self.best_x[18:24].copy()
            else:
                cat_ids = self.best_x[18:24].copy()
                
            x_final = np.concatenate([best_c, cat_ids])
            self._evaluate(x_final, func)
            
        # Hessian-Regularized Trust-Region Exploitation
        if self._hess_func is not None and self.evals < self.budget:
            H = self._hess_func(self.best_x)
            eigs, Q = np.linalg.eigh(H)
            H_reg = Q @ np.diag(np.abs(eigs) + 1e-4) @ Q.T
            
            def local_obj(xc):
                xc = np.clip(xc, -1.0, 1.0)
                x_full = np.concatenate([xc, self.best_x[18:24]])
                return self._evaluate(x_full, func)
                
            res = minimize(local_obj, self.best_x[:18], hess=lambda xc: H_reg, 
                           method='trust-constr', bounds=[(-1.0, 1.0)]*18, 
                           options={'maxiter': 25, 'verbose': 0})
            
            if self.evals < self.budget:
                cand = np.concatenate([res.x, self.best_x[18:24]])
                cand_f = self._evaluate(cand, func)
                if cand_f < self.best_f:
                    self.best_f = cand_f
                    self.best_x = cand.copy()

        return self.best_f, self.best_x