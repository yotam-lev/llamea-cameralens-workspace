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
        self.__dict__.update(tool.__class__.__dict__)
        self.sample = tool.sample
        self.latin_hypercube_sampling = tool
        self.lhs = tool

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._tool, name)

class LensOptimisation(Problem):
    def __init__(self, training_instances=None, test_instances=None, budget_factor: int = 5000, 
                 name: str = "Lensoptimisation", eval_timeout: int = 6000, **kwargs):
        if training_instances is None: training_instances = [(seed,) for seed in range(1, 10)]
        if test_instances is None: test_instances = [(seed,) for seed in range(11, 16)]
        
        super().__init__(
            training_instances=training_instances, test_instances=test_instances,
            name=name, eval_timeout=eval_timeout,
            dependencies=["cma>=3.3", "pydoe>=0.3"],
            imports="import scipy\nimport numpy as np\nimport jax.numpy as jnp\nfrom scipy.optimize import minimize\n"
        )
        self.budget_factor = budget_factor
        self.task_prompt = (
            "### STRICT CODING STANDARDS ###\n"
            "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for the best solution and `es.result[1]` for the best fitness. NEVER use `es.xbest`, `es[0]`, or `es.best.x`.\n"
            "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize`. The solution is in `res.x`. Ensure `x0` is a 1D array.\n"
            "3. SCOPING: Define all logic within the `Optimizer` class. If you use helper methods, they MUST accept `func` and `grad_func` as arguments explicitly.\n"
            "4. DIMENSIONS: indices [0-17] are continuous curvatures/distances. indices [18-23] are categorical glass IDs.\n\n"
            "### TASK ###\n"
            "Minimize a 24D lens loss function. indices [0-17] are geometric (differentiable), [18-23] are glass IDs.\n"
            "Framework FORCE-INJECTS 'self.func' and 'self.grad_func' into your Optimizer. Use them.\n"
            "The helper 'lhs' is available via global name or import.\n"
        )
        self.example_prompt = "class Optimizer:\n    def __init__(self, budget, dim):\n        self.budget=budget; self.dim=dim\n    def __call__(self, func, grad_func):\n        return self.func(np.zeros(self.dim)), np.zeros(self.dim)\n"
        self.format_prompt = "# Description: <Provide a concise description of the algorithm, limited to a maximum of two sentences.>\n# Code:\n```python\n<code>\n```"

    def _get_sandbox_env(self):
        import scipy.optimize, cma, random, math
        import jax.numpy as jnp
        lhs_tool = LHSWrapper()
        safe_env = {
            "__builtins__": __builtins__, "np": np, "numpy": np, "scipy": scipy,
            "minimize": scipy.optimize.minimize, "cma": cma, "math": math, "random": random,
            "jnp": jnp, "sys": sys, "types": types, "latin_hypercube_sampling": lhs_tool, "lhs": lhs_tool,
        }
        for name in ["latin_hypercube_sampling", "lhs"]:
            sys.modules[name] = CallableModule(name, lhs_tool)
            safe_env[name] = sys.modules[name]
        return safe_env

    def _build_objective(self):
        from pathlib import Path
        workspace_root = Path(__file__).resolve().parent
        for _ in range(5):
            if (workspace_root / "camera-lens-simulation").exists(): break
            workspace_root = workspace_root.parent
        simulation_path = workspace_root / "camera-lens-simulation"
        if str(simulation_path) not in sys.path: sys.path.insert(0, str(simulation_path))
        from examples.double_gauss_objective import DoubleGaussObjective
        obj = DoubleGaussObjective(enable_grad=True)
        def func(x): return obj.objective_theta(np.clip(x, *obj.bounds()))
        def grad_func(x):
            xc, xi = obj.split_theta(np.clip(x, *obj.bounds()))
            return obj.gradient_cont_int(xc, xi)
        return func, grad_func, obj.n_theta, *obj.bounds()

    def evaluate(self, solution: Solution, test=False, ioh_dir="") -> Solution:
        try:
            func, grad_fn, dim, lb, ub = self._build_objective()
            exec_env = self._get_sandbox_env()
            # Clean LLM code of conflicting imports
            clean_code = re.sub(r'^(?:from|import)\s+(?:latin_hypercube_sampling|lhs).*$', '', solution.code, flags=re.MULTILINE)
            
            # Inject safety globals for common LLM hallucinations
            exec_env['population_size'] = 20
            exec_env['pop_size'] = 20
            exec_env['glass_ids'] = list(range(100))
            
            exec(clean_code, exec_env)
            
            # Flexible Class Extraction
            OptimizerClass = exec_env.get("Optimizer")
            if not OptimizerClass:
                # If the LLM named the class something else, find any class with a __call__ method
                from .lens_optimisation import LHSWrapper # ignore this internal class
                for name, val in exec_env.items():
                    if isinstance(val, type) and val is not LHSWrapper and hasattr(val, "__call__"):
                        OptimizerClass = val
                        break
            
            if not OptimizerClass: return solution.set_scores(-np.inf, feedback="No valid 'Optimizer' class found.")

            def call_optimizer(opt_inst, f, g, env):
                # Double-Layer Injection
                opt_inst.func, opt_inst.grad_func = f, g
                env['func'], env['grad_func'] = f, g 
                sig = inspect.signature(opt_inst.__call__)
                return opt_inst(f, g) if len(sig.parameters) >= 2 else opt_inst(f)

            try:
                # PURE POSITIONAL INIT
                sig_init = inspect.signature(OptimizerClass.__init__)
                dry_run_opt = OptimizerClass(10, dim) if len(sig_init.parameters) >= 3 else OptimizerClass(10)
                call_optimizer(dry_run_opt, lambda x: float(np.sum(x**2)), lambda x: 2*x[:18], exec_env)
            except Exception as e:
                return solution.set_scores(-np.inf, feedback=f"Optimizer failed during initial dry run: {e}")

            instances = self.test_instances if test else self.training_instances
            losses = []
            scale = (ub - lb) / 2.0
            for (seed,) in instances:
                np.random.seed(seed)
                sig_init = inspect.signature(OptimizerClass.__init__)
                opt = OptimizerClass(self.budget_factor, dim) if len(sig_init.parameters) >= 3 else OptimizerClass(self.budget_factor)
                b_f = lambda xn: func(lb + (xn + 1.0) / 2.0 * (ub - lb))
                b_g = lambda xn: grad_fn(lb + (xn + 1.0) / 2.0 * (ub - lb)) * scale[:18]
                best_f, _ = call_optimizer(opt, b_f, b_g, exec_env)
                losses.append(float(best_f))
            solution.set_scores(-np.mean(losses), feedback=f"Mean loss: {np.mean(losses):.6f}")
        except Exception as e:
            solution.set_scores(-np.inf, feedback=f"Error: {e}")
        return solution

    def test(self, solution: Solution) -> Solution:
        orig = self.training_instances; self.training_instances = self.test_instances
        result = self.evaluate(solution, test=True)
        self.training_instances = orig; return result

    def to_dict(self): return {"name": self.name, "budget_factor": self.budget_factor}
