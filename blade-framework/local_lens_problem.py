"""
Importable LensOptimisation subclass that reuses the current
interpreter and ensures CameraLensSimulation is on PYTHONPATH
for the evaluation subprocess.
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


class LocalLensOptimisation(LensOptimisation):
    """Reuse the current interpreter instead of creating a temp venv."""

    def _ensure_env(self):
        if self._env_path is not None:
            return
        self._env_path = Path(tempfile.mkdtemp(prefix="blade_env_"))
        self._python_bin = Path(sys.executable)