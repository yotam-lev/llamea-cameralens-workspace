"""
BLADE Problem wrapper for the CameraLensSimulation Double-Gauss
lens optimisation benchmark.

Each BLADE Solution contains LLM-generated Python code that defines
a callable optimizer class. This Problem executes that optimizer
against the DoubleGaussObjective and scores it by the best loss found.
"""
from __future__ import annotations

import traceback
import numpy as np
import builtins
import os
from ..problem import Problem
from ..solution import Solution
#from ..utils import OverBudgetException, aoc_logger, correcct_aoc




class LensOptimisation(Problem):
    """
    Problem class for Double-Gauss camera lens design.
    
    LLM generates a Python class with signature:
        class Optimizer:
            def __init__(self, budget: int, dim: int): ...
            def __call__(self, func) -> tuple[float, np.ndarray]: ...
    
    Where:
        - func(x) returns a scalar loss for parameter vector x
        - budget is the max number of func evaluations allowed
        - dim is the dimensionality of the search space
    """

    def __init__(
        self,
        training_instances=None,
        test_instances=None,
        budget_factor: int = 5000,
        name: str = "Lensoptimisation",
        eval_timeout: int = 600,
        seeds=5,
        logger=None,
        dependencies=None,
        imports=None,
    ):
        # Dependencies that will be pip-installed in the eval sandbox
        if dependencies is None:
            dependencies = [
                "jax>=0.4",
                "jaxlib>=0.4",
                "pandas>=2",
                "openpyxl>=3",
            ]
            try:
                dependencies.append("lensgopt")  
            except ImportError:
                dependencies.append("lensgopt @ git+https://github.com/yotam-lev/CameraLensSimulation.git")
        if imports is None:
            imports = (
                "import scipy\n",
                "import math\n",
                "import numpy as np\n"
                "import jax.numpy as jnp\n"
                "from scipy.optimize import minimize, differential_evolution\n"
                "from lensgopt import DoubleGaussObjective\n"
            )

        # Training instances = (seed,) tuples for reproducibility
        if training_instances is None:
            training_instances = [(seed,) for seed in range(1, 6)]
        if test_instances is None:
            test_instances = [(seed,) for seed in range(6, 16)]

        super().__init__(
            logger=logger,
            training_instances=training_instances,
            test_instances=test_instances,
            name=name,
            eval_timeout=eval_timeout,
            dependencies=dependencies,
            imports=imports,
        )

        self.budget_factor = budget_factor

        # ---- Prompts for the LLM ----
        self.task_prompt = (
            "You are tasked with writing a novel black-box optimisation "
            "algorithm to minimize a camera lens design loss function. "
            "The loss function is non-convex, ~24-dimensional (18 continuous "
            "parameters: 10 curvatures + 8 distances, plus 6 discrete glass "
            "material IDs treated as continuous for this task), and combines "
            "RMS spot size across multiple wavelengths and field angles with "
            "penalty terms for focal length, thickness, and working distance "
            "constraints.\n\n"
            "The search space has box bounds. The objective is to MINIMIZE "
            "the loss value (lower is better). Your optimizer will receive a "
            "callable `func(x)` that takes a numpy array and returns a scalar "
            "loss.\n\n"
        )

        self.example_prompt = (
            "Write a Python class named `Optimizer` with:\n"
            "```python\n"
            "import numpy as np\n\n"
            "class Optimizer:\n"
            "    def __init__(self, budget: int, dim: int):\n"
            "        self.budget = budget\n"
            "        self.dim = dim\n\n"
            "    def __call__(self, func) -> tuple[float, np.ndarray]:\n"
            "        # func(x) returns scalar loss for x in R^dim\n"
            "        best_f = float('inf')\n"
            "        best_x = None\n"
            "        for _ in range(self.budget):\n"
            "            x = np.random.uniform(-1, 1, self.dim)\n"
            "            f = func(x)\n"
            "            if f < best_f:\n"
            "                best_f = f\n"
            "                best_x = x\n"
            "        return best_f, best_x\n"
            "```\n\n"
        )

        self.format_prompt = (
            """
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""
        )

        # EoH compatibility settings
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

    def _build_objective(self):
        """
        Lazily construct the DoubleGaussObjective.
        Returns (objective_fn, dim, bounds_lb, bounds_ub).
        """
        from examples.double_gauss_objective import DoubleGaussObjective

        obj = DoubleGaussObjective(
            enable_grad=False, enable_hessian=False
        )
        lb, ub = obj.bounds()
        dim = obj.n_theta

        def func(x):
            """Wrapper: numpy array → scalar loss."""
            x_clipped = np.clip(x, lb, ub)
            return obj.objective_theta(x_clipped)

        return func, dim, lb, ub

    def evaluate(self, solution: Solution) -> Solution:
        """
        Execute the LLM-generated optimizer code on training instances.
        """
        try:
            # Build the objective
            func, dim, lb, ub = self._build_objective()
            budget = self.budget_factor

            # Compile and instantiate the LLM-generated optimizer
            safe_globals = {"__builtins__": builtins, "np": np, "numpy": np}
            exec(solution.code, safe_globals)
            OptimizerClass = safe_globals.get("Optimizer")
            if OptimizerClass is None:
                solution.set_scores(
                    -np.inf,
                    feedback="No class named 'Optimizer' found in the code.",
                    error="Missing Optimizer class.",
                )
                return solution

            # Evaluate across training instances (random seeds)
            losses = []
            for (seed,) in self.training_instances:
                np.random.seed(seed)
                optimizer = OptimizerClass(budget=budget, dim=dim)

                # Create a bounded wrapper so the optimizer sees [-1, 1]
                def bounded_func(x_normalized):
                    x_real = lb + (x_normalized + 1.0) / 2.0 * (ub - lb)
                    return func(x_real)

                best_f, best_x = optimizer(bounded_func)
                losses.append(float(best_f))

            mean_loss = np.mean(losses)
            # BLADE maximizes fitness; lens optimisation minimizes loss
            # So fitness = -mean_loss (higher fitness = lower loss = better)
            fitness = -mean_loss

            feedback = (
                f"Mean loss across {len(self.training_instances)} seeds: "
                f"{mean_loss:.6f}. Best single run: {min(losses):.6f}."
            )
            solution.set_scores(fitness, feedback=feedback)

        except Exception as e:
            solution.set_scores(
                -np.inf,
                feedback=f"Error during evaluation: {e}",
                error=e,
            )

        return solution

    def test(self, solution: Solution) -> Solution:
        """Final evaluation on held-out test instances."""
        # Swap instances temporarily
        orig = self.training_instances
        self.training_instances = self.test_instances
        result = self.evaluate(solution)
        self.training_instances = orig
        return result

    def to_dict(self):
        return {
            "name": self.name,
            "budget_factor": self.budget_factor,
            "eval_timeout": self.eval_timeout,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
        }