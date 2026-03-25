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
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            logger.info("[config] 🟢 NVIDIA GPU detected: %s", gpu_info)
            _PLATFORM_CACHE = "nvidia_gpu"
            return _PLATFORM_CACHE
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for Apple Silicon
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


def _get_nvidia_vram_gb() -> float:
    """Return total VRAM in GB of the first NVIDIA GPU, or 0."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi reports in MiB
            mib = float(result.stdout.strip().split("\n")[0])
            return mib / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


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
        1. Gemini API (if GEMINI_API_KEY is set and a test query succeeds)
        2. Ollama local fallback (qwen2.5-coder:14b)

    Returns:
        LLM: An instance of Gemini_LLM or Ollama_LLM.
    """
    api_key = os.getenv("GEMINI_API_KEY")
     # Force Ollama fallback for now, as Gemini API is not yet available

    if api_key:
        try:
            from iohblade.llm import Gemini_LLM

            llm = Gemini_LLM(api_key=api_key, model="gemini-2.5-flash")


            # Test with a minimal query to catch 429 / invalid key
            response = llm.query(
                [{"role": "user", "content": "Respond with only: OK"}]
            )
            if response:
                logger.info("[config] ✅ Using Gemini API (gemini-2.5-flash)")
                return llm

        except Exception as e:
            logger.warning("[config] ⚠️  Gemini failed: %s", e)

    from iohblade.llm import Ollama_LLM

    logger.info("[config] 🔄 Using Ollama local (qwen2.5-coder:14b)")
    return Ollama_LLM("qwen2.5-coder:14b")