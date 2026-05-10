
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
import time

def test_venv_creation_speed():
    """
    Test how long it takes to create a virtual environment and install dependencies.
    """
    # Simulate what _ensure_env does
    import virtualenv
    
    start = time.time()
    env_dir = tempfile.mkdtemp(prefix="blade_env_test_")
    print(f"Created temp dir: {env_dir}")
    
    virtualenv.cli_run([env_dir])
    python_bin = Path(env_dir) / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    # Minimal dependencies to test speed
    deps = ["numpy"]
    
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", *deps],
        check=True,
        capture_output=True,
        text=True,
    )
    
    end = time.time()
    print(f"Environment setup took: {end - start:.2f} seconds")
    
    shutil.rmtree(env_dir)
    print("Cleaned up.")

if __name__ == "__main__":
    test_venv_creation_speed()
