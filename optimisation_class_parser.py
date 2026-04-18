class OptimisationClassParser:
    """
    A utility class to parse and separate lines from a string containing 
    optimization class code.
    """
    def __init__(self, optclass: str):
        """
        Initialize the parser with the raw class code string.
        
        Args:
            optclass (str): The whole class string with lines separated by \n.
        """
        self.optclass = optclass

    def get_separated_lines(self) -> list[str]:
        """
        Returns the whole class with each line separated into a list.
        """
        return self.optclass.split('\n')

    def print_separated_lines(self):
        """
        Outputs the whole class with each line printed separately.
        """
        lines = self.get_separated_lines()
        for i, line in enumerate(lines, 1):
            print(f"{line}")

if __name__ == "__main__":
    # Example usage:
    example_code ="import numpy as np\nfrom scipy.stats import qmc\nfrom scipy.optimize import minimize\nimport cma\n\ndef lhs(n_samples, n_dim):\n    sampler = qmc.LatinHypercube(d=n_dim)\n    sample = sampler.random(n=n_samples)\n    return qmc.scale(sample, np.full(n_dim, -1), np.full(n_dim, 1))\n\nclass Optimizer:\n    def __init__(self, budget: int, dim: int):\n        self.budget = budget\n        self.dim = dim\n        self.evals = 0\n        self.best_f = float('inf')\n        self.best_x = np.zeros(dim)\n        self.cma_es = None\n        self.cma_pop_size = 50\n        self.tr_radius = 0.1\n        self.tr_count = 0\n        self.tr_max_iter = 5\n        self.tr_min_radius = 1e-6\n        self.tr_adapt_rate = 0.1\n\n    def _evaluate(self, x, func):\n        if self.evals >= self.budget: return float('inf')\n        # FORCE ROUNDING OF CATEGORICALS TO NEAREST 0.5 in [-1, 1]\n        eval_x = x.copy()\n        eval_x[18:24] = np.round(eval_x[18:24] * 2) / 2 \n        eval_x[18:24] = np.clip(eval_x[18:24], -1.0, 1.0)\n        f = func(eval_x)\n        self.evals += 1\n        if f < self.best_f:\n            self.best_f = f\n            self.best_x = eval_x.copy()\n        return f\n\n    def _trust_region_step(self, x_center, func, grad_func, radius):\n        \"\"\"Perform trust-region step using L-BFGS-B\"\"\"\n        x_disc = x_center[18:24]\n        \n        def cost_wrap(x_cont):\n            return func(np.concatenate([x_cont, x_disc]))\n        \n        def grad_wrap(x_cont):\n            grad24 = grad_func(np.concatenate([x_cont, x_disc]))\n            return grad24[:18]\n        \n        # Use bounds for continuous variables\n        bounds = [(-1, 1)] * 18\n        \n        # Start from current point\n        x_start = x_center[:18].copy()\n        \n        # Perform local optimization\n        try:\n            res = minimize(cost_wrap, x_start, method='L-BFGS-B', \n                          jac=grad_wrap, bounds=bounds, options={'maxiter': 20})\n            x_local = np.concatenate([res.x, x_disc])\n            return x_local\n        except:\n            # If optimization fails, return the center\n            return x_center.copy()\n\n    def __call__(self, func, grad_func=None):\n        # 1. Initialization (LHS)\n        pop = lhs(n_samples=20, n_dim=self.dim)\n        for x in pop: \n            self._evaluate(x, func)\n\n        # 2. Initialize CMA-ES for continuous variables\n        # Only initialize once\n        if self.cma_es is None:\n            x0 = self.best_x.copy()\n            # Only continuous variables for CMA-ES\n            x0_cont = x0[:18]\n            self.cma_es = cma.CMAEvolutionStrategy(x0_cont, 0.2, {'popsize': self.cma_pop_size})\n        \n        # 3. Main Loop\n        while self.evals < self.budget:\n            # Generate new population using CMA-ES\n            X = self.cma_es.ask()\n            fitness = []\n            \n            for x in X:\n                if self.evals >= self.budget: break\n                \n                # Combine continuous and discrete parts\n                x_full = np.zeros(self.dim)\n                x_full[:18] = x\n                x_full[18:24] = self.best_x[18:24]  # Keep best discrete variables for now\n                \n                # Evaluate\n                f = self._evaluate(x_full, func)\n                fitness.append(f)\n                \n                if self.evals >= self.budget: break\n            \n            if self.evals >= self.budget: break\n            \n            # Tell CMA-ES about the fitness\n            self.cma_es.tell(X, fitness)\n            \n            # Occasionally perform local search on best solution\n            if grad_func is not None and np.random.rand() < 0.2:\n                x_best = self.best_x.copy()\n                x_disc = x_best[18:24]\n                \n                # Perform trust-region optimization\n                x_local = self._trust_region_step(x_best, func, grad_func, self.tr_radius)\n                \n                if self.evals >= self.budget: break\n                self._evaluate(x_local, func)\n                \n                # Update trust region radius based on success\n                if np.random.rand() < 0.5:  # Randomly adjust radius\n                    self.tr_radius = max(self.tr_min_radius, self.tr_radius * 0.9)\n                else:\n                    self.tr_radius = min(0.5, self.tr_radius * 1.1)\n        \n        return self.best_f, self.best_x"
    
    parser = OptimisationClassParser(example_code)
    print("--- Separated Lines ---")
    parser.print_separated_lines()
