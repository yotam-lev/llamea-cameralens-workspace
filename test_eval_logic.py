
import numpy as np
import builtins
import scipy
from scipy.optimize import differential_evolution, minimize

def evaluate(code):
    try:
        safe_globals = {"__builtins__": builtins, "np": np, "numpy": np}
        exec(code, safe_globals)
        OptimizerClass = safe_globals.get("Optimizer")
        
        def mock_func(x):
            return np.sum(x**2)
            
        opt = OptimizerClass(budget=10, dim=2)
        best_f, best_x = opt(mock_func)
        return f"Success: {best_f}"
    except Exception as e:
        return f"Error: {e}"

code = """
import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[float, np.ndarray]:
        bounds = [(-1, 1)] * self.dim
        result_de = differential_evolution(func, bounds, popsize=min(10, self.budget), maxiter=1)
        return result_de.fun, result_de.x
"""

print(evaluate(code))
