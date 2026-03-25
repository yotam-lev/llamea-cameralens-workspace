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
            "You are tasked with writing a high-performance, mixed-variable black-box "
            "optimization algorithm to minimize a complex camera lens design loss function.\n\n"
   
            "## Problem Structure\n"
            "- 24-dimensional search space in normalized [-1, 1] bounds.\n"
            "- 18 continuous variables: 10 surface curvatures and 8 inter-element distances.\n"
            "- 6 categorical variables: Glass material IDs (mapped to integers). Note: These "
            "IDs are unordered; nearby IDs do NOT correspond to similar optical properties.\n"
            "- The loss is a weighted sum of RMS spot size across 3 wavelengths and multiple " 
            "field angles, plus non-linear geometric penalty terms.\n\n"
   
            "## Landscape Properties\n"
            "- Extremely non-convex and multimodal: Thousands of deep local optima exist.\n"
            "- Sharp 'Death Penalty' boundaries: Infeasible geometric configurations (e.g., "
            "lenses overlapping) return extremely high loss values, creating cliff-like edges.\n"
            "- Mixed-Variable Mismatch: The 6D glass subspace is purely categorical, while "
            "the 18D geometry subspace is smooth and differentiable within feasible basins.\n\n"

            "## Effective Strategies\n"
            "- Memetic Refinement: Use a global metaheuristic (e.g., Differential Evolution) "
            "to find basins, then trigger local search (e.g., CMA-ES) on the 18D continuous "
            "subspace only.\n"
            "- Categorical Handling: Do not treat glass IDs as continuous; use discrete "
            "operators (mutation/crossover) or speciation for those dimensions.\n"
            "- Constraint-Aware Sampling: Use biased Latin Hypercube Sampling or repair "
            "mechanisms to stay within the narrow feasible manifold of valid lens geometry.\n"
            "- Diversity Preservation: Implement crowding or niching to prevent the population "
            "from collapsing into a single glass combination too early.\n\n"

            "The objective is to MINIMIZE the loss value. Your optimizer receives a "
            "callable `func(x)` that takes a numpy array of shape (24,) and returns a scalar.\n\n"

            "Environment Note: The following modules are already imported and available "
            "in the global namespace: `np`, `scipy`, `math`, `cma`, `latin_hypercube_sampling`, "
        )