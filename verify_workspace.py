import iohblade
import lensgopt
import numpy as np

print(f"iohblade version: {iohblade.__version__}")
print(f"lensgopt version: {lensgopt.__version__}")

try:
    from lensgopt.optics.lens import Lens
    print("Successfully imported Lens from lensgopt.optics.lens")
except ImportError as e:
    print(f"Failed to import Lens: {e}")

try:
    from iohblade.problem import Problem
    print("Successfully imported Problem from iohblade.problem")
except ImportError as e:
    print(f"Failed to import Problem: {e}")

print("Workspace verification complete.")
