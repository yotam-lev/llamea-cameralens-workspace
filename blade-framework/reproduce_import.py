import os
import sys

# Simulation of what LensOptimisation does
CAMERA_LENS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "camera-lens-simulation")
)
if CAMERA_LENS_ROOT not in sys.path:
    sys.path.insert(0, CAMERA_LENS_ROOT)

try:
    from examples.double_gauss_objective import DoubleGaussObjective
    print("Successfully imported DoubleGaussObjective")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
