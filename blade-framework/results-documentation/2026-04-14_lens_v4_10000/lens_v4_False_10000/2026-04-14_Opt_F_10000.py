import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import cma

def lhs(n_samples, n_dim):
    sampler = qmc.LatinHypercube(d=n_dim)
    sample = sampler.random(n=n_samples)
    return qmc.scale(sample, [0]*n_dim, [1]*n_dim)

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.es = None
        self.trust_region_radius = 0.5
        self.min_trust_region_radius = 0.01

    def _evaluate(self, x, func):
        if self.evals >= self.budget: return float('inf')
        # FORCE ROUNDING OF CATEGORICALS TO NEAREST 0.5 in [-1, 1]
        eval_x = x.copy()
        eval_x[18:24] = np.round(eval_x[18:24] * 2) / 2 
        eval_x[18:24] = np.clip(eval_x[18:24], -1.0, 1.0)
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def _local_search(self, x_cont, x_disc, func, grad_func):
        """Perform local search using L-BFGS-B with trust region constraints"""
        def cost_wrap(x_cont):
            return func(np.concatenate([x_cont, x_disc]))
        
        def grad_wrap(x_cont):
            return grad_func(np.concatenate([x_cont, x_disc]))[:18]
        
        # Define bounds for continuous variables
        bounds = [(-1, 1)] * 18
        
        # Perform local search
        res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=bounds)
        
        if res.success:
            x_new = np.concatenate([res.x, x_disc])
            return x_new
        return None

    def __call__(self, func, grad_func=None):
        # 1. Initialization (LHS)
        pop = lhs(n_samples=10, n_dim=self.dim)
        for x in pop: self._evaluate(x, func)
        
        # 2. Initialize CMA-ES with best solution
        self.es = cma.CMAEvolutionStrategy(self.best_x.copy(), 0.3)
        
        # 3. Main Loop
        while self.evals < self.budget:
            # CMA-ES generation
            X = self.es.ask()
            F = []
            for x in X:
                f = self._evaluate(x, func)
                F.append(f)
            self.es.tell(X, F)
            
            # Local search with L-BFGS-B on continuous variables if gradient info available
            if grad_func is not None and self.evals < self.budget:
                # Fix discrete variables
                x_disc = self.best_x[18:24]
                x_cont = self.best_x[:18]
                
                # Perform local search
                x_local = self._local_search(x_cont, x_disc, func, grad_func)
                if x_local is not None:
                    self._evaluate(x_local, func)
            
            # Update trust region based on recent improvement
            if self.evals > 10:
                # Reduce trust region if no significant improvement
                if self.evals % 50 == 0:
                    self.trust_region_radius = max(self.min_trust_region_radius, 
                                                   self.trust_region_radius * 0.95)
        
        return self.best_f, self.best_x