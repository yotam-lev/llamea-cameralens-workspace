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
    example_code = "import numpy as np\nfrom scipy.stats import qmc\nfrom scipy.optimize import minimize\nimport random\n\ndef lhs(n_samples, n_dim):\n    sampler = qmc.LatinHypercube(d=n_dim)\n    sample = sampler.random(n=n_samples)\n    return qmc.scale(sample, np.full(n_dim, -1), np.full(n_dim, 1))\n\nclass Optimizer:\n    def __init__(self, budget: int, dim: int):\n        self.budget = budget\n        self.dim = dim\n        self.evals = 0\n        self.best_f = float('inf')\n        self.best_x = np.zeros(dim)\n        self.de_pop = None\n        self.de_pop_size = 200\n        self.trust_region_radius = 0.1\n        self.min_trust_radius = 0.01\n        self.trust_radius_decay = 0.95\n        self.discrete_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])\n        self.de_history = []\n\n    def _evaluate(self, x, func):\n        if self.evals >= self.budget: return float('inf')\n        f = func(x)\n        self.evals += 1\n        if f < self.best_f:\n            self.best_f = f\n            self.best_x = x.copy()\n        return f\n\n    def __call__(self, func, grad_func=None):\n        # 1. Initialization (LHS)\n        pop = lhs(n_samples=20, n_dim=self.dim)\n        for x in pop: self._evaluate(x, func)\n        \n        # 2. Initialize DE population for continuous variables\n        # Create initial population for DE\n        de_pop = []\n        for _ in range(self.de_pop_size):\n            individual = np.zeros(self.dim)\n            individual[:18] = np.random.uniform(-1, 1, 18)\n            individual[18:24] = np.random.choice(self.discrete_values, 6)\n            de_pop.append(individual)\n        self.de_pop = np.array(de_pop)\n        \n        # 3. Main Loop\n        while self.evals < self.budget:\n            # Phase 1: DE for global exploration of continuous variables\n            if self.de_pop is not None and self.evals < self.budget:\n                new_pop = []\n                for i in range(self.de_pop_size):\n                    if self.evals >= self.budget: break\n                    \n                    # DE mutation and crossover\n                    idxs = random.sample(range(self.de_pop_size), 3)\n                    a, b, c = self.de_pop[idxs[0]], self.de_pop[idxs[1]], self.de_pop[idxs[2]]\n                    \n                    # Differential evolution mutation\n                    F = 0.8\n                    mutant = a + F * (b - c)\n                    mutant = np.clip(mutant, -1, 1)\n                    \n                    # Crossover with current individual\n                    crossover_rate = 0.7\n                    trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.de_pop[i])\n                    \n                    # Ensure discrete variables stay in valid range\n                    trial[18:24] = np.clip(trial[18:24], -1.0, 1.0)\n                    \n                    # Evaluate trial individual\n                    f_trial = self._evaluate(trial, func)\n                    new_pop.append(trial)\n                \n                # Replace old population with new population\n                self.de_pop = np.array(new_pop)\n            \n            # Phase 2: Trust-region local refinement with L-BFGS-B\n            if grad_func is not None and self.evals < self.budget:\n                # Get current best solution\n                x_disc = self.best_x[18:24].copy()\n                x_cont = self.best_x[:18].copy()\n                \n                def cost_wrap(x_cont):\n                    return func(np.concatenate([x_cont, x_disc]))\n                \n                def grad_wrap(x_cont):\n                    return grad_func(np.concatenate([x_cont, x_disc]))[:18]\n                \n                # Use L-BFGS-B for local refinement with trust region\n                res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18)\n                x_cont_new = res.x\n                x_full_new = np.concatenate([x_cont_new, x_disc])\n                f_new = self._evaluate(x_full_new, func)\n                \n                # Update best solution if improved\n                if f_new < self.best_f:\n                    self.best_x[:18] = x_cont_new\n                    # Shrink trust region after successful step\n                    self.trust_region_radius = max(self.min_trust_radius, self.trust_region_radius * self.trust_radius_decay)\n                else:\n                    # Expand trust region after unsuccessful step\n                    self.trust_region_radius = min(1.0, self.trust_region_radius * 1.1)\n            \n            # Phase 3: Discrete mutation for categorical variables\n            if self.evals < self.budget:\n                # Create a new candidate by mutating the best solution's discrete variables\n                x_disc = self.best_x[18:24].copy()\n                # Apply discrete mutation: randomly change one or more discrete variables\n                num_mutations = np.random.randint(1, 4)  # Mutate 1-3 variables\n                for _ in range(num_mutations):\n                    if self.evals >= self.budget: break\n                    idx = np.random.randint(0, 6)\n                    x_disc[idx] = np.random.choice(self.discrete_values)\n                \n                x_full = np.concatenate([self.best_x[:18], x_disc])\n                f = self._evaluate(x_full, func)\n                if f < self.best_f:\n                    self.best_x[18:24] = x_disc.copy()\n        \n        return self.best_f, self.best_x"
    
    parser = OptimisationClassParser(example_code)
    print("--- Separated Lines ---")
    parser.print_separated_lines()
