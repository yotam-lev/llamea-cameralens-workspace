import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[float, np.ndarray]:
        # Initialize population with random solutions within bounds [-1, 1]
        population_size = 50
        population = np.random.uniform(-1, 1, (population_size, self.dim))
        
        # Evaluate initial population
        fitnesses = np.array([func(x) for x in population])
        best_idx = np.argmin(fitnesses)
        best_f = fitnesses[best_idx]
        best_x = population[best_idx]

        # Evolutionary Strategy with mutations and elitism, improved by adding adaptive mutation rate
        mutation_rate = 0.1
        for _ in range(self.budget - population_size):
            # Select a parent solution randomly
            parent_idx = np.random.randint(0, population_size)
            parent = population[parent_idx]
            
            # Mutate the parent to create a new candidate with adaptive mutation rate
            candidate = parent + mutation_rate * np.random.randn(self.dim)
            candidate = np.clip(candidate, -1, 1)  # Ensure bounds
            
            # Evaluate the candidate
            candidate_f = func(candidate)
            
            # Update best solution if the candidate is better
            if candidate_f < best_f:
                best_f = candidate_f
                best_x = candidate
            
            # Replace the worst solution in the population if the candidate is better
            if candidate_f < np.max(fitnesses):
                worst_idx = np.argmax(fitnesses)
                population[worst_idx] = candidate
                fitnesses[worst_idx] = candidate_f
                
            # Adaptive mutation rate based on progress
            if _ % 10 == 0:
                improvement_rate = (best_f - np.min(fitnesses)) / np.abs(np.min(fitnesses))
                mutation_rate *= 0.95 + 0.05 * improvement_rate

        return best_f, best_x