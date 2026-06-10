import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.swarm_size = 60
        self.w = 0.7
        self.c1 = 1.4
        self.c2 = 1.4
        self.lambda_h = 0.1
        self.hess_freq = 25
        self.lso_freq = 15
        self.mut_rate = 0.05

    def _evaluate(self, x, func):
        if self.evals >= self.budget:
            return float('inf')
        # Strict boundary and categorical enforcement
        eval_x = np.clip(x.copy(), -1.0, 1.0)
        eval_x[18:24] = np.clip(np.round(eval_x[18:24]), 0, 5).astype(int)
        
        f = func(eval_x)
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = eval_x.copy()
        return f

    def _regularize_hessian(self, H):
        eigvals, eigvecs = np.linalg.eigh(H)
        # Guarantee positive definiteness via absolute eigenvalues
        eigvals = np.abs(eigvals) + 1e-6
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def __call__(self, func, grad_func=None, hess_func=None, **kwargs):
        # 1. Initialization via LHS
        n_init = min(20, max(1, self.budget // 20))
        X = np.random.uniform(-1, 1, size=(self.swarm_size, self.dim))
        
        # Initialize velocities
        V = np.zeros_like(X)
        
        # Store continuous and categorical states separately for specialized handling
        X_cont = X[:, :18].copy()
        X_cat = np.round(np.clip(X[:, 18:24], 0, 5)).astype(int)
        
        # Best positions
        pbest_X = X.copy()
        pbest_f = np.full(self.swarm_size, float('inf'))
        
        # Evaluate initial swarm
        for i in range(self.swarm_size):
            x_full = np.concatenate([X_cont[i], X_cat[i]])
            f = self._evaluate(x_full, func)
            pbest_f[i] = f
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_full

        gen = 0
        hess_counter = 0
        lso_counter = 0

        while self.evals < self.budget:
            # Check budget at start of generation
            if self.evals >= self.budget:
                break

            # 2. Continuous Subspace Update (Hessian-Preconditioned PSO)
            # Compute gradient for global best to drive Newton step
            # Only compute Hessians periodically to save budget
            grad_step = np.zeros(18)
            if hess_func is not None and hess_counter % self.hess_freq == 0:
                hess_counter = 0
                # Safety check before Hessian evaluation
                if self.evals >= self.budget: break
                
                # Construct full vector for Hessian query
                full_best = np.concatenate([self.best_x[:18], self.best_x[18:24]])
                hess = hess_func(full_best)
                
                if grad_func is not None:
                    if self.evals >= self.budget: break
                    grad = grad_func(full_best)[:18]
                    
                    # Regularize and solve
                    H_reg = self._regularize_hessian(hess)
                    grad_step = -np.linalg.solve(H_reg, grad)
                else:
                    grad_step = np.zeros(18)
            hess_counter += 1

            gbest_cont = self.best_x[:18].copy()

            for i in range(self.swarm_size):
                # PSO Update
                r1 = np.random.rand(18)
                r2 = np.random.rand(18)
                
                v_update = (self.w * V[i] + 
                            self.c1 * r1 * (pbest_X[i, :18] - X_cont[i]) + 
                            self.c2 * r2 * (gbest_cont - X_cont[i]))
                
                # Inject Hessian preconditioned step (Damping for stability)
                V[i] = v_update + self.lambda_h * grad_step
                X_cont[i] += V[i]

            # Clip continuous bounds
            X_cont = np.clip(X_cont, -1.0, 1.0)

            # 3. Categorical Update (Discrete Adaptive Mutation)
            # Direct integer update to avoid drift issues
            for i in range(self.swarm_size):
                for j in range(6):
                    if np.random.rand() < self.mut_rate:
                        # Mutation: jump to random valid ID
                        X_cat[i, j] = np.random.randint(0, 6)
                    elif np.random.rand() < 0.02:
                        # Local mutation: increment/decrement
                        step = np.random.choice([-1, 1])
                        X_cat[i, j] = np.clip(X_cat[i, j] + step, 0, 5)

            # 4. Evaluation and Repair
            for i in range(self.swarm_size):
                x_full = np.concatenate([X_cont[i], X_cat[i]])
                f = self._evaluate(x_full, func)
                
                # Update personal best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_X[i] = x_full

                # Update global best
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x_full
                    gbest_cont = x_full[:18].copy()
                    # Update velocity direction towards new global best immediately
                    V[i] += self.c2 * (gbest_cont - X_cont[i])

            # 5. Periodic Trust-Region Refinement (Memetic)
            lso_counter += 1
            if lso_counter % self.lso_freq == 0 and self.evals < self.budget - 10:
                if self.evals >= self.budget: break
                
                # Refine continuous subspace of global best
                res = minimize(
                    lambda xc: func(np.concatenate([xc, self.best_x[18:]])),
                    self.best_x[:18],
                    jac=lambda xc: grad_func(np.concatenate([xc, self.best_x[18:]])) if grad_func else None,
                    bounds=[(-1, 1)] * 18,
                    method='trust-constr',
                    options={'maxiter': 10, 'verbose': 0}
                )
                if res.success and res.fun < self.best_f:
                    self._evaluate(np.concatenate([res.x, self.best_x[18:]]), func)

            gen += 1

        return self.best_f, self.best_x