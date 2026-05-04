import numpy as np
from scipy.spatial.distance import cdist
import cma

class LensOptimizer(Optimizer):
    def __init__(self, budget, dim=24):         
        super().__init__(budget, dim)          
        self.cont_dims = 18        
        self.cat_dims = 6
        self.cat_bounds = [0, 10]  # Glass IDs from 0 to 10 (inclusive)

    def __call__(self, func, grad_func=None):
        # Initialize population with biased Latin Hypercube Sampling        
        pop = self._initialize_population()
        
        best_x = None        
        best_val = float('inf')
        t = 0
        
        while t < self.budget:
            # DE selection and mutation
            trial_pop = []
            for i in range(len(pop)):
                a, b, c = np.random.choice(len(pop), 3, replace=False)
                mutant = pop[a] + 0.8 * (pop[b] - pop[c])
                
                # Crossover: mix continuous and categorical parts
                cross_prob = 0.7
                for j in range(self.dim):
                    if j < self.cont_dims:
                        if np.random.rand() > cross_prob:
                            mutant[j] = pop[i][j]
                    else:
                        # For categorical variables, use discrete recombination
                        if np.random.rand() > cross_prob:
                            mutant[j] = pop[i][j]

                # Repair invalid geometries (e.g., overlapping lenses)
                if not self._is_valid_geometry(mutant):
                    mutant = self._repair_geometry(mutant)

                trial_pop.append(mutant)
            
            # Evaluate and select
            for i in range(len(pop)):
                val = func(trial_pop[i])
                if val < best_val:
                    best_val = val
                    best_x = trial_pop[i].copy()
                
                if val < func(pop[i]):
                    pop[i] = trial_pop[i]
            
            # Local search on continuous subspace every 50 iterations
            if t % 50 == 0 and t > 0:
                x_cont = pop[0][:self.cont_dims]
                es = cma.CMAEvolutionStrategy(x_cont, 0.2)
                while not es.stop():
                    X = es.ask()
                    fitnesses = [func(np.concatenate([x, pop[0][self.cont_dims:]])) for x in X]
                    es.tell(X, fitnesses)
                
                best_cont = es.result.xbest
                best_x[:self.cont_dims] = best_cont
                best_val = func(best_x)
            
            t += 1
            
        return best_val, best_x

    def _initialize_population(self):
        # Use biased Latin Hypercube Sampling for better coverage
        n_pop = max(20, self.budget // 10)
        pop = []
        
        # Generate samples in continuous space with bias towards valid regions
        for i in range(n_pop):
            x = np.zeros(self.dim)
            
            # Continuous variables: curvatures and distances (normalized [-1, 1])
            x[:self.cont_dims] = np.random.uniform(-1, 1, self.cont_dims)
            
            # Categorical variables: glass IDs
            for j in range(self.cat_dims):
                x[self.cont_dims + j] = np.random.randint(self.cat_bounds[0], self.cat_bounds[1]+1)
                
            pop.append(x)
        return pop

    def _is_valid_geometry(self, x):
        # Simple geometric feasibility check (example: no overlapping lenses)
        # This is a placeholder - real implementation would be more complex
        distances = x[self.cont_dims:self.cont_dims+8]
        return np.all(distances > 0.01)  # Placeholder constraint

    def _repair_geometry(self, x):
        # Repair invalid geometry by clamping or resampling
        x[self.cont_dims:self.cont_dims+8] = np.clip(x[self.cont_dims:self.cont_dims+8], 0.01, 1)
        return x