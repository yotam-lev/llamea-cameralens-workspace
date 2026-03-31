"""
LensOptimisation with domain-specific context injected into prompts.
"""
import os
import sys
import tempfile
from pathlib import Path

CAMERA_LENS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "camera-lens-simulation")
)
if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)
_current = os.environ.get("PYTHONPATH", "")
if CAMERA_LENS_ROOT not in _current:
    os.environ["PYTHONPATH"] = CAMERA_LENS_ROOT + os.pathsep + _current

from iohblade.problems.lens_optimisation import LensOptimisation


class ContextualLensOptimisation(LensOptimisation):
    """Injects optics domain knowledge into the LLM prompts."""

    def __init__(self, **kwargs):
        # 1. Override the imports string before passing it to the parent class
        #    so LLaMEA doesn't inject the broken import into the LLM's code.
        if "imports" not in kwargs:
            kwargs["imports"] = (
                "import scipy\n"
                "import math\n"
                "import numpy as np\n"
                "import jax.numpy as jnp\n"
                "from scipy.optimize import minimize, differential_evolution\n"
            )

        super().__init__(**kwargs)

        self.task_prompt = (
            "### STRICT CODING STANDARDS ###\n"
            "1. CMA-ES ACCESS: When using `cma.CMAEvolutionStrategy`, use `es.result[0]` for the best solution and `es.result[1]` for the best fitness. NEVER use `es.xbest`, `es[0]`, or `es.best.x`.\n"
            "2. SCIPY MINIMIZE: Use `scipy.optimize.minimize(func, x0, jac=grad_func, ...)`. The solution is in `res.x`. Ensure `x0` is a 1D array.\n"
            "3. SCOPING: Define all logic within the `Optimizer` class. If you use helper methods, they MUST accept `func` and `grad_func` as arguments explicitly.\n"
            "4. DIMENSIONS: indices [0-17] are continuous curvatures/distances. indices [18-23] are categorical glass IDs.\n\n"
            "### ENVIRONMENT NOTE ###\n"
            "The following modules are already imported and available in the global namespace: `np`, `scipy`, `math`, `cma`, `latin_hypercube_sampling` (alias `lhs`).\n"
            "Write a completely self-contained Python class named exactly Optimizer. Do not inherit from any base classes.\n"
            "The `grad_func(x)` provided to `__call__` returns an 18-dimensional vector containing "
            "the gradients for the first 18 continuous variables only. For the categorical variables, "
            "the gradient is undefined (effectively 0).\n\n"
            "### LENS DESIGN TASK ###\n"
            "You are an elite algorithm designer and optical physicist specializing in meta-heuristics, "
            "hybrid optimization, and non-convex landscapes. Your task is to write a novel, highly efficient "
            "optimization algorithm to minimize a camera lens design loss function.\n\n"
            "### Problem Physics & Landscape:\n"
            "The objective calculates RMS spot size via ray-tracing alongside physical penalty constraints. "
            "The landscape is highly multimodal, extremely non-convex, and filled with infeasible 'cliffs' "
            "where rays fail to trace. \n\n"
            "### Search Space (`dim = 24`):\n"
            "The entire search space is normalized to strictly bounded `[-1, 1]` box constraints.\n"
            "- Indices `x[0:18]`: 18 Continuous parameters (lens curvatures and physical thicknesses).\n"
            "- Indices `x[18:24]`: 6 Categorical glass material IDs (evaluated as continuous `[-1, 1]` inputs under the hood).\n\n"
            "### The Gradient Advantage (`grad0_cont`):\n"
            "You are provided with `grad0_cont` (numpy array, shape `(18,)`). This is the exact analytical gradient "
            "of the 18 continuous parameters at the standard baseline lens design. A brilliant algorithm will use this "
            "to break symmetry—such as using it to take an initial pseudo-gradient step, biasing the initial population "
            "distribution, or guiding local-search mutations for the continuous variables while using global search for the glass IDs.\n\n"
            "### Execution Constraints:\n"
            "1. You MUST strictly adhere to the `budget` limit (max evaluations of `func`).\n"
            "2. The objective is to MINIMIZE the loss (lower is better).\n"
            "3. Return the absolute `best_f` and `best_x` found.\n\n"
        )
        