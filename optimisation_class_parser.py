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
    example_code = "python\nclass Optimizer:\n    def __init__(self, budget, dim):\n        self.budget = budget\n        self.dim = dim\n\n    def __call__(self, func, grad_func):\n        # Initialize the best solution and fitness with infinity and None\n        best_f = float('inf')\n        best_x = None\n        \n        # Generate an initial population using Latin Hypercube Sampling\n        initial_population = lhs(self.dim, samples=10)\n        \n        # Bias the initial population distribution based on the gradient information\n        for i in range(initial_population.shape[0]):\n            # Use the gradient to adjust the first 18 continuous variables\n            initial_population[i, :18] += 0.05 * grad0_cont\n        \n        # Initialize CMA-ES with the biased initial population and adaptive parameters\n        es = cma.CMAEvolutionStrategy(initial_population.mean(axis=0), 0.3)\n        \n        for _ in range(self.budget):\n            solutions = []\n            fitness_values = []\n            \n            # Generate a new batch of solutions\n            while len(solutions) < 10:\n                x = es.ask()\n                f = func(x)\n                \n                if f < best_f:\n                    best_f = f\n                    best_x = x\n                \n                solutions.append(x)\n                fitness_values.append(f)\n            \n            # Tell CMA-ES the results of the evaluations, allowing it to adapt its parameters\n            es.tell(solutions, fitness_values)\n        \n        return best_f, best_x\n"
    
    parser = OptimisationClassParser(example_code)
    print("--- Separated Lines ---")
    parser.print_separated_lines()
