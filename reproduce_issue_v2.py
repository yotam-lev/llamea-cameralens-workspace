
import numpy as np
import builtins
import scipy
from scipy.optimize import differential_evolution, minimize

code = """
import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[float, np.ndarray]:
        bounds = [(-1, 1)] * self.dim
        # Testing if differential_evolution is in the scope of this method
        result_de = differential_evolution(func, bounds, popsize=min(10, self.budget), maxiter=1)
        return result_de.fun, result_de.x
"""

# Mirroring LensOptimisation.evaluate's safe_globals
safe_globals = {"__builtins__": builtins, "np": np, "numpy": np}
try:
    print("Executing code...")
    exec(code, safe_globals)
    print("differential_evolution in safe_globals?", "differential_evolution" in safe_globals)
    
    OptimizerClass = safe_globals.get("Optimizer")
    print(f"OptimizerClass found: {OptimizerClass}")
    
    def mock_func(x):
        return np.sum(x**2)
        
    opt = OptimizerClass(budget=10, dim=2)
    print("Calling optimizer...")
    best_f, best_x = opt(mock_func)
    print(f"Success: {best_f}, {best_x}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
