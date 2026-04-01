"""
BLADE Problem wrapper for the CameraLensSimulation Double-Gauss
lens optimisation benchmark.
"""
from __future__ import annotations

import traceback
import numpy as np
import re
import sys
import types
import inspect
from scipy.stats import qmc
from ..problem import Problem
from ..solution import Solution


class OverBudgetExecption (Exception):
    """"Raised when the optimizer exceeds the allowed budget.w"""

class LHSWrapper:
    """Polymorphic tool that handles almost any calling pattern and arg naming."""
    def __call__(self, n_samples, n_dim=None, **kwargs):
        # Resolve 'dim' vs 'n_dim' hallucination
        actual_dim = n_dim if n_dim is not None else kwargs.get('dim')
        if actual_dim is None:
            # Fallback for positional confusion
            if isinstance(n_samples, int) and isinstance(n_dim, int):
                pass # Already correct
            elif len(kwargs) > 0:
                actual_dim = list(kwargs.values())[0]
        
        sampler = qmc.LatinHypercube(d=actual_dim if actual_dim else 24)
        return sampler.random(n=n_samples) * 2 - 1
    
    def sample(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
        
    def __getattr__(self, name):
        """Redirects any sub-attribute (e.g. .LatinHypercube()) to itself."""
        return self

class CallableModule(types.ModuleType):
    """A module that behaves like a function to support 'import lhs; lhs()'."""
    def __init__(self, name, tool):
        super().__init__(name)
        self._tool = tool
        self.sample = tool.sample
        self.latin_hypercube_sampling = tool
        self.lhs = tool

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._tool, name)

class Optimizer:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
    def __call__(self, func, grad_func):
        raise NotImplementedError("Subclasses must implement __call__")

class LensOptimisation(Problem):
    def __init__(
            self, 
            training_instances=None, 
            test_instances=None, 
            budget_factor: 
            int = 5000, 
            name: str = "Lensoptimisation", 
            eval_timeout: int = 6000, 
            seeds=5,
            task_prompt: str = None,       
            example_prompt: str = None,    
            format_prompt: str = None,
            logger=None,
            **kwargs):
        if training_instances is None: training_instances = [(seed,) for seed in range(1, 10)]
        if test_instances is None: test_instances = [(seed,) for seed in range(11, 16)]
        
        super().__init__(
            training_instances=training_instances, test_instances=test_instances,
            name=name, eval_timeout=eval_timeout,
            dependencies=["cma>=3.3", "pydoe>=0.3"],
            imports="import scipy\nimport numpy as np\nimport jax.numpy as jnp\nfrom scipy.optimize import minimize\n"
        )
        self.budget_factor = budget_factor
        self.task_prompt = task_prompt if task_prompt is not None else (
            "### STRICT CODING STANDARDS ###\n"
            "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for the best solution and `es.result[1]` for the best fitness. NEVER use `es.xbest`, `es[0]`, or `es.best.x`. To get the population size, use `es.popsize` (NOT population_size).\n"
            "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`. Ensure `x0` is a 1D array.\n"
            "3. SCOPING: Define all logic within the `Optimizer` class. If you use helper methods, they MUST accept `func` and `grad_func` as arguments explicitly.\n"
            "4. DIMENSIONS: indices [0-17] are continuous curvatures/distances. indices [18-23] are categorical glass IDs.\n"
            "5. GRADIENT SHAPE AWARENESS: `grad0_cont` is shape (18,). If you apply it to a 24D vector, you MUST slice the target array first to prevent broadcast errors (e.g., `x[:18] += 0.1 * self.grad0_cont`).\n\n"
            "### TASK ###\n"
            "Minimize a 24D lens loss function. indices [0-17] are geometric (differentiable), [18-23] are glass IDs.\n"
            "Framework FORCE-INJECTS 'self.func' and 'self.grad_func' into your Optimizer. Use them.\n"
            "The helper 'lhs' is available via global name or import.\n"

        )
        
       
        self.example_prompt = example_prompt if example_prompt is not None else(
            "class Optimizer:\n"
            "    def __init__(self, budget, dim, grad0_cont):\n"
            "        self.budget = budget\n"
            "        self.dim = dim\n"
            "        self.grad0_cont = grad0_cont # YOU MUST SAVE THIS\n"
            "    def __call__(self, func, grad_func):\n"
            "        best_f = float('inf')\n"
            "        best_x = None\n"
            "        # ALWAYS use keyword arguments for lhs:\n"
            "        initial_population = lhs(n_samples=20, n_dim=self.dim)\n"
            "        ...\n"
        )
        self.format_prompt = format_prompt if format_prompt is not None else (
            "# Description: <Provide a concise description of the algorithm, limited to a maximum of two sentences.>\n# Code:\n```python\n<code>\n```"
        )
    



    def _get_sandbox_env(self):
        import scipy.optimize, cma, random, math
        import jax.numpy as jnp
        lhs_tool = LHSWrapper()
        safe_env = {
            "__builtins__": __builtins__, "np": np, "numpy": np, "scipy": scipy,
            "minimize": scipy.optimize.minimize, "cma": cma, "math": math, "random": random,
            "jnp": jnp, "sys": sys, "types": types, "latin_hypercube_sampling": lhs_tool, "lhs": lhs_tool,
            "Optimizer": Optimizer,
        }
        for name in ["latin_hypercube_sampling", "lhs"]:
            sys.modules[name] = CallableModule(name, lhs_tool)
            safe_env[name] = sys.modules[name]
        return safe_env

    def _build_objective(self):
        from pathlib import Path

        print(f"this is the path {Path}")
        workspace_root = Path(__file__).resolve().parent
        for _ in range(5):
            if (workspace_root / "camera-lens-simulation").exists(): break
            workspace_root = workspace_root.parent
        simulation_path = workspace_root / "camera-lens-simulation"
        if str(simulation_path) not in sys.path: 
            sys.path.insert(0, str(simulation_path))
        from examples.double_gauss_objective import DoubleGaussObjective
        obj = DoubleGaussObjective(
            enable_grad=True, enable_hessian=False
            )
        
        print(f"this is the double gauss object {obj} \n" )
        
        lb, ub = obj.bounds()
        print(f"OBjective bounds {obj.bounds()}\n")
        dim = obj.n_theta
        x0_cont, x0_ids = obj.init_from_templates()
        print(f"x0 continuous {x0_cont}, x0 ids {x0_ids} \n")
        grad0_cont = obj.gradient_cont_int(x0_cont, x0_ids)

        def func(x): 
            x_clipped = np.clip(x, lb, ub)

            print(f"This is the clipped x function thing {x_clipped}\n")
            return obj.objective_theta(x_clipped)
            
        def grad_fn(x):
            x_clipped = np.clip(x, lb, ub)
            xc, xi = obj.split_theta(x_clipped)
            return obj.gradient_cont_int(xc, xi)

        return func, grad_fn, dim, lb, ub, grad0_cont

    def evaluate(self, solution: Solution) -> Solution:
        print("We have entered the eval class \n")
        """
        Execute the LLM-generated optimizer code on training instances.
        """
        try:
            func, grad_fn, dim, lb, ub, grad0_cont = self._build_objective()
            print(f"func{func}, grad_fn: {grad_fn}, lb{lb}, ub{ub}, grad0_cont{grad0_cont}")
            budget = self.budget_factor
            print(f"budget {budget}")
            exec_env = self._get_sandbox_env() 

            
            clean_code = re.sub(r'^(?:from|import)\s+(?:latin_hypercube_sampling|lhs).*$', '', solution.code, flags=re.MULTILINE)
            
            exec_env['population_size'] = 20
            exec_env['pop_size'] = 20
            exec_env['glass_ids'] = list(range(100))

            
            exec(clean_code, exec_env)
  
            # 1. ROBUST CLASS EXTRACTION
            # We look for a subclass of Optimizer that isn't Optimizer itself
            OptimizerClass = exec_env.get("Optimizer")
            
            # 2. Safety Fallback (Just in case the LLM names it MyOptimizer, Solver, etc.)
            if not OptimizerClass:
                ignore_list = ["LHSWrapper", "DoubleGaussObjective", "Solution", "Problem"]
                for name, val in exec_env.items():
                    if isinstance(val, type) and name not in ignore_list:
                        # Check if it has an execution method
                        if any(hasattr(val, m) for m in ["optimize", "solve", "run", "minimize", "__call__"]):
                            OptimizerClass = val
                            break
            
            if not OptimizerClass: 
                return solution.set_scores(-np.inf, feedback="No valid 'Optimizer' class found.")

            sig_init = inspect.signature(OptimizerClass.__init__)
            init_params = sig_init.parameters

            def create_optimizer(b, d, g_cont):
                kwargs = {}
                if 'budget' in init_params: kwargs['budget'] = b
                if 'dim' in init_params: kwargs['dim'] = d
                if 'grad0_cont' in init_params: kwargs['grad0_cont'] = g_cont
                
                if not kwargs:
                    num_args = len(init_params) - 1 
                    if num_args >= 3: return OptimizerClass(b, d, g_cont)
                    if num_args == 2: return OptimizerClass(b, d)
                    return OptimizerClass(b)
                return OptimizerClass(**kwargs)

            # 2. ROBUST EXECUTION WITH ENTRY POINT HUNTING
            def call_optimizer(opt_inst, f, g, env):
                opt_inst.func, opt_inst.grad_func = f, g
                env['func'], env['grad_func'] = f, g 
                
                entry_methods = ["__call__", "optimize", "solve", "run", "minimize"]
                last_error = None
                
                for method_name in entry_methods:
                    if hasattr(opt_inst, method_name):
                        method = getattr(opt_inst, method_name)
                        try:
                            sig = inspect.signature(method)
                            num_params = len(sig.parameters)
                            # Handle both (func, grad_func) and (func)
                            if num_params >= 2: 
                                return method(f, g) 
                            else:
                                return method(f)
                        except NotImplementedError as e:
                            last_error = e
                            continue 
                            
                if last_error: raise last_error 
                raise AttributeError(f"No valid execution method found. Expected one of: {entry_methods}")

            # Dry Run Phase
            try:
                dry_run_opt = create_optimizer(10, dim, grad0_cont)
                mock_func = lambda x: float(np.sum(x**2))
                mock_grad = lambda x: 2*x[:18]
                call_optimizer(dry_run_opt, mock_func, mock_grad, exec_env)
            except Exception as e:
                return solution.set_scores(-np.inf, feedback=f"Optimizer failed during initial dry run: {e}")

            # 3. Production Evaluation Loop
            losses = []
            scale = (ub - lb) / 2.0
            for (seed,) in self.training_instances:
                np.random.seed(seed)
                opt = create_optimizer(self.budget_factor, dim, grad0_cont)
                
                def bounded_func(xn):
                    xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
                    return func(xr)
                    
                def bounded_grad(xn):
                    xr = lb + (xn + 1.0) / 2.0 * (ub - lb)
                    # Chain rule: d f(xr(xn)) / d xn = df/dxr * dxr/dxn = grad * (ub-lb)/2
                    return grad_fn(xr) * scale[:18]
                
                best_f, best_x = call_optimizer(opt, bounded_func, bounded_grad, exec_env)
                losses.append(float(best_f))
                
            mean_loss = np.mean(losses)
            solution.set_scores(-mean_loss, feedback=f"Mean loss: {mean_loss:.6f}. Best single run: {min(losses):.6f}.")
            
        except Exception as e:
            solution.set_scores(-np.inf, feedback=f"Error during evaluation: {e}")
            
        return solution

    def test(self, solution: Solution) -> Solution:
        orig = self.training_instances; self.training_instances = self.test_instances
        result = self.evaluate(solution)
        self.training_instances = orig; return result

    def to_dict(self): return {"name": self.name, "budget_factor": self.budget_factor}
