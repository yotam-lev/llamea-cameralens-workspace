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

    def _ensure_env(self):
        if self._env_path is not None:
            return
        self._env_path = Path(tempfile.mkdtemp(prefix="blade_env_"))
        self._python_bin = Path(sys.executable)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task_prompt = (
            "You are tasked with writing a novel black-box optimisation "
            "algorithm to minimise a camera lens design loss function.\n\n"

            "## Problem Structure\n"
            "- 24-dimensional search space in normalised [-1, 1] bounds.\n"
            "- The real parameters are: 10 surface curvatures (continuous), "
            "8 inter-element distances (continuous), 6 glass material IDs "
            "(integer, but treated as continuous — nearby IDs do NOT "
            "correspond to similar materials).\n"
            "- The loss combines RMS spot size across 3 wavelengths and "
            "multiple field angles, plus penalty terms.\n\n"

            "## Landscape Properties\n"
            "- Highly multimodal: many local optima.\n"
            "- Mixed continuous-discrete: 6 of 24 dimensions are effectively "
            "categorical. Consider treating them differently.\n"
            "- Many infeasible regions that return very high loss values.\n"
            "- Smooth within feasible basins — gradient-free methods like "
            "CMA-ES, Differential Evolution, and Nelder-Mead work well "
            "for local refinement.\n\n"

            "## Effective Strategies\n"
            "- Latin Hypercube Sampling for diverse initialisation.\n"
            "- Differential Evolution with adaptive parameters (F, CR).\n"
            "- CMA-ES for local refinement of promising solutions.\n"
            "- Two-phase: global exploration then local exploitation.\n"
            "- Restart strategies when stuck in local optima.\n"
            "- Keep population diverse — avoid premature convergence.\n\n"

            "The objective is to MINIMISE the loss value (lower is better). "
            "Your optimiser receives a callable `func(x)` that takes a "
            "numpy array of shape (dim,) and returns a scalar loss.\n\n"

            "Environment Note: The following modules are already imported "
            "and available in the global namespace: `np`, `scipy`, `math`, and "
            "`optics`. Do NOT write import statements for these."
        )