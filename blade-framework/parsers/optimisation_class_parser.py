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
    example_code = "import numpy as np\n\nclass Optimizer:\n    def __init__(self, budget: int, dim: int):\n        self.budget = budget\n        self.dim = dim\n        self.evals = 0\n        self.best_f = float('inf')\n        self.best_x = np.zeros(dim)\n        self.grad0_cont = None\n\n    def set_initial_gradient(self, grad0_cont):\n        self.grad0_cont = grad0_cont\n\n    def _evaluate(self, x, func):\n        \"\"\"Wrapper to safely track budget and update best solution.\"\"\"\n        if self.evals >= self.budget:\n            return float('inf')\n        f = func(x)\n        self.evals += 1\n        if f < self.best_f:\n            self.best_f = f\n            self.best_x = x.copy()\n        return f\n\n    def __call__(self, func, grad_func=None):\n        # Initialization (LHS)\n        initial_population = lhs(n_samples=20, n_dim=self.dim)\n        for x in initial_population:\n            self._evaluate(x, func)\n\n        # Initial gradient step\n        if grad_func is not None and self.grad0_cont is not None:\n            baseline_x = np.zeros(self.dim)\n            lr = 0.01  # Learning rate for the gradient step\n            gradient_step = lr * self.grad0_cont[:18]\n            baseline_x[:18] -= gradient_step\n            self._evaluate(baseline_x, func)\n\n        # Enhanced Random Search with Gaussian mutation \n        while self.evals < self.budget:\n            if np.random.rand() < 0.5:\n                x = np.random.uniform(-1, 1, self.dim)\n            else:\n                # Gaussian mutation around the best found solution\n                sigma = 0.1  # Mutation strength\n                x = self.best_x.copy()\n                x[:18] += np.random.normal(0, sigma, 18)  # Mutate continuous variables\n                # Ensure bounds are respected for continuous variables\n                x[:18] = np.clip(x[:18], -1, 1)\n                # Categorical variables remain unchanged as they are discrete\n\n            self._evaluate(x, func)\n\n        return self.best_f, self.best_x"
    
    parser = OptimisationClassParser(example_code)
    print("--- Separated Lines ---")
    parser.print_separated_lines()
