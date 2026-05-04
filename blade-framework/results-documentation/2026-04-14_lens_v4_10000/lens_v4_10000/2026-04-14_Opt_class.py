import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import random

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)
        self.de_pop = []
        self.de_pop_size = 200
        self.de_iter = 0

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

    def _de_mutation(self, pop, idx, F=0.8):
        """DE mutation strategy"""
        a, b, c = random.sample(range(len(pop)), 3)
        while a == idx or b == idx or c == idx:
            a, b, c = random.sample(range(len(pop)), 3)
        return pop[a] + F * (pop[b] - pop[c])

    def _de_crossover(self, target, mutant, CR=0.7):
        """DE crossover"""
        trial = np.copy(target)
        if random.random() < CR:
            j = random.randint(0, len(target) - 1)
            trial[j] = mutant[j]
        else:
            # Ensure at least one dimension is mutated
            j = random.randint(0, len(target) - 1)
            trial[j] = mutant[j]
        return trial

    def __call__(self, func, grad_func=None):
        # 1. Initialization (LHS)
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        pop = sampler.random(n=20)
        pop = qmc.scale(pop, [0]*self.dim, [1]*self.dim)
        # Convert to [-1, 1] bounds
        pop = pop * 2 - 1
        for x in pop: 
            self._evaluate(x, func)

        # 2. Initialize DE population
        self.de_pop = [self.best_x.copy()]
        for _ in range(self.de_pop_size - 1):
            # Generate random individuals in bounds
            x = np.random.uniform(-1, 1, self.dim)
            self.de_pop.append(x)
        self.de_pop = np.array(self.de_pop)

        # 3. Main Loop
        while self.evals < self.budget:
            # DE generation
            new_pop = []
            for i in range(len(self.de_pop)):
                if self.evals >= self.budget: break
                
                # Mutation
                mutant = self._de_mutation(self.de_pop, i)
                
                # Crossover
                trial = self._de_crossover(self.de_pop[i], mutant)
                
                # Ensure bounds
                trial = np.clip(trial, -1, 1)
                
                # Evaluate
                f = self._evaluate(trial, func)
                new_pop.append(trial)
                
                # Keep the best individual from DE
                if f < self.best_f:
                    self.de_pop[i] = trial.copy()
                else:
                    # Keep the old individual
                    pass
            
            if len(new_pop) > 0:
                self.de_pop = np.array(new_pop)
            
            # Local refinement with L-BFGS-B
            if grad_func is not None and self.evals < self.budget:
                # Use best solution from DE for local search
                x_disc = self.best_x[18:24]
                x_cont = self.best_x[:18]
                
                def cost_wrap(x_cont):
                    return func(np.concatenate([x_cont, x_disc]))
                
                def grad_wrap(x_cont):
                    return grad_func(np.concatenate([x_cont, x_disc]))[:18]
                
                try:
                    res = minimize(cost_wrap, x_cont, method='L-BFGS-B', jac=grad_wrap, bounds=[(-1, 1)]*18)
                    if res.success:
                        x_new = np.concatenate([res.x, x_disc])
                        self._evaluate(x_new, func)
                except:
                    pass

        return self.best_f, self.best_x