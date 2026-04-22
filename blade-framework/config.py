"""
Shared configuration for all experiment scripts.

Provides:
    - get_llm()      : LLM selection (Gemini API → Ollama fallback)
    - get_n_jobs()   : Parallel worker count based on detected hardware
    - get_platform() : Cached platform string
    - PLATFORM       : "nvidia_gpu", "apple_silicon", or "cpu_only" (lazy, backward-compat)
"""
import os
import logging
import multiprocessing
import platform
import subprocess
import sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Platform Detection ────────────────────────────────────────────

_PLATFORM_CACHE: str | None = None


def _detect_platform() -> str:
    """
    Detect hardware by checking for NVIDIA GPU or Apple Silicon.
    Result is cached after the first call so subprocess calls run exactly once.

    Returns:
        "nvidia_gpu"     — NVIDIA GPU detected (HPC / workstation)
        "apple_silicon"  — Apple M-series chip detected (MacBook)
        "cpu_only"       — neither detected
    """
    global _PLATFORM_CACHE
    if _PLATFORM_CACHE is not None:
        return _PLATFORM_CACHE

    # Check for NVIDIA GPU via nvidia-smi
    try:
        import jax
        if jax.default_backend() == "gpu":
            devices = jax.devices("gpu")
            device_names = [d.device_kind for d in devices]
            logger.info("[config] 🟢 JAX GPU backend active. Devices: %s", device_names)
            _PLATFORM_CACHE = "nvidia_gpu"
            return _PLATFORM_CACHE
    except Exception as e:
        logger.debug("[config] JAX GPU check bypassed or failed: %s", e)

    # 2. Fallback: Check for NVIDIA GPU via nvidia-smi 
    # (Triggers if JAX isn't installed with CUDA support, but hardware is present)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            logger.warning("[config] ⚠️ NVIDIA GPU hardware detected (%s), but JAX might be using CPU. Check your jaxlib CUDA installation.", gpu_info)
            _PLATFORM_CACHE = "nvidia_gpu"
            return _PLATFORM_CACHE
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = _get_apple_chip_name()
        logger.info("[config] 🍎 Apple Silicon detected: %s", chip)
        _PLATFORM_CACHE = "apple_silicon"
        return _PLATFORM_CACHE

    logger.info("[config] ⚪ No GPU detected, CPU-only mode")
    _PLATFORM_CACHE = "cpu_only"
    return _PLATFORM_CACHE


def get_platform() -> str:
    """Return the (cached) platform string."""
    return _detect_platform()


def __getattr__(name: str):
    """Lazy module-level attribute for backward-compatible ``PLATFORM`` access."""
    if name == "PLATFORM":
        return get_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_apple_chip_name() -> str:
    """Return the Apple chip model (e.g. 'Apple M2 Pro')."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Apple Silicon (unknown model)"


# ── Compute Configuration ─────────────────────────────────────────

def get_n_jobs() -> int:
    """
    Return the number of parallel workers based on the platform.

    - NVIDIA GPU (HPC): respect SLURM allocation if available,
      otherwise use all cores minus 2.
    - Apple Silicon (MacBook): cap at 4 to avoid thermal throttling
      and leave headroom for Ollama.
    - CPU-only: half of available cores.

    Returns:
        int: Number of parallel workers for Experiment(n_jobs=...).
    """
    total_cores = multiprocessing.cpu_count()
    current_platform = get_platform()

    if current_platform == "nvidia_gpu":
        # Respect SLURM allocation if running as a cluster job
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            n = int(slurm_cpus)
            logger.info("[config] 🖥️  HPC: %d workers (SLURM allocation)", n)
            return max(1, n)

        slurm_ntasks = os.getenv("SLURM_NTASKS")
        if slurm_ntasks:
            n = int(slurm_ntasks)
            logger.info("[config] 🖥️  HPC: %d workers (SLURM_NTASKS)", n)
            return max(1, n)

        # No SLURM — use most cores, leave 2 for system
        n = max(1, total_cores - 2)
        logger.info("[config] 🖥️  NVIDIA workstation: %d workers (%d cores)", n, total_cores)
        return n

    if current_platform == "apple_silicon":
        # M-series: efficiency + performance cores
        # Cap at 4 to avoid thermal throttling and leave room for Ollama
        n = min(4, max(1, total_cores // 2))
        logger.info("[config] 💻 Apple Silicon: %d workers (%d cores)", n, total_cores)
        return n

    # cpu_only fallback
    n = max(1, total_cores // 2)
    logger.info("[config] 💻 CPU-only: %d workers (%d cores)", n, total_cores)
    return n


# ── LLM Selection ─────────────────────────────────────────────────

def get_llm():
    """
    Returns a ready-to-use LLM instance.

    Priority:
        1. Rotating Gemini API (if GEMINI_API_KEY_1 and GEMINI_API_KEY_2 are set)
        2. Single Gemini API (if GEMINI_API_KEY is set)
        3. Ollama local fallback
    """
    #key1 = os.getenv("GEMINI_API_KEY_1")
    #key2 = os.getenv("GEMINI_API_KEY_2")
    #single_key = os.getenv("GEMINI_API_KEY")
    key1 = None
    key2 = key1
    single_key = key1

    from iohblade.llm import Gemini_LLM, Multi_LLM, RateLimited_LLM

    # Scenario 1: Dual Keys
    if key1 and key2:
        try:
            llm1 = Gemini_LLM(api_key=key1, model="gemini-2.0-flash")
            llm2 = Gemini_LLM(api_key=key2, model="gemini-2.0-flash")
            
            # Combine them to alternate strictly
            multi = Multi_LLM([llm1, llm2])
            
            # Wrap with rate limiting: 3 calls/min PER key = 6 total per minute
            limited = RateLimited_LLM(multi, calls_per_minute=6)
            
            logger.info("[config] ✅ Using Rotating Gemini API (Dual Keys, 6/pm total)")
            return limited
        except Exception as e:
            logger.warning("[config] ⚠️  Dual Gemini setup failed: %s", e)

    # Scenario 2: Single Key
    api_key = single_key or key1 or key2
    if api_key:
        try:
            llm = Gemini_LLM(api_key=api_key, model="gemini-2.0-flash")
            # Limit to 3 calls/min
            limited = RateLimited_LLM(llm, calls_per_minute=3)
            logger.info("[config] ✅ Using Gemini API (Single Key, 3/pm)")
            return limited
        except Exception as e:
            logger.warning("[config] ⚠️  Gemini setup failed: %s", e)

    # Fallback to Ollama
    from iohblade.llm import Ollama_LLM
    
    try:
        result = subprocess.run(
            ["ollama", "ls"],
            capture_output=True, text=True, check=True,
        )
        lines = result.stdout.strip().split('\n')
        
        if len(lines) < 2:
            print("No models found. Please pull a model first using 'ollama pull <model>'.")
            sys.exit(1)
            
        # Parse the models, skipping the header line
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0]) # The first column is the model name
                
        # Print the enumerated list
        print("Available models:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")
            
    except FileNotFoundError:
        print("Error: Ollama is not installed or not found in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error communicating with Ollama: {e}")
        sys.exit(1)

    while True:
        choice = input("\nEnter the number of the model you want to use: ").strip()
        
        try:
            choice_idx = int(choice) - 1
            
            # Check if the number is within our list bounds
            if 0 <= choice_idx < len(models):
                return Ollama_LLM(models[choice_idx])
            else:
                print(f"Invalid selection. Please choose a number between 1 and {len(models)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    

    