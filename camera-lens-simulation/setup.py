"""
setup.py  –  Build-time script for the lensgopt C++ extension (hillvallimpl).

The C++ extension is OPTIONAL.  Set the environment variable
    LENSGOPT_BUILD_HILLVALL=1
before running  `pip install -e .`  or  `uv sync`  to compile it.

Without that variable the package installs as pure-Python (JAX ray
tracing, loss functions, parsers, visualization all work without C++).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
HILLVALLEA_SRC = ROOT / "external" / "HillVallEA" / "HillVallEA"
HILLVALLEA_BINDINGS_SRC = (
    ROOT / "lensgopt" / "optimisation" / "hillvall" / "hillvall_bindings.cpp"
)

# ---------------------------------------------------------------------------
# Opt-in gate:  only build the C++ extension when explicitly requested.
# ---------------------------------------------------------------------------
BUILD_HILLVALL = os.environ.get("LENSGOPT_BUILD_HILLVALL", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _relative_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _hillvall_sources() -> list[str]:
    """Collect all .cpp files from the HillVallEA submodule + the bindings."""
    hillvallea_cpp = sorted(Path(p) for p in glob.glob(str(HILLVALLEA_SRC / "*.cpp")))
    return [_relative_to_root(HILLVALLEA_BINDINGS_SRC)] + [
        _relative_to_root(p) for p in hillvallea_cpp
    ]


def _cxx_compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]


# ---------------------------------------------------------------------------
# Custom build_ext that auto-initialises the git submodule
# ---------------------------------------------------------------------------
class BuildExtWithSubmodules(build_ext):
    """
    Extended ``build_ext`` that ensures the HillVallEA git submodule is
    checked out before compiling.
    """

    def run(self) -> None:
        # Only do submodule / source work when there are extensions to build.
        if self.extensions:
            self._ensure_hillvallea_sources()
            self._refresh_hillvall_extension_sources()
        super().run()

    # -- submodule helpers --------------------------------------------------

    def _ensure_hillvallea_sources(self) -> None:
        if HILLVALLEA_SRC.exists() and any(HILLVALLEA_SRC.glob("*.cpp")):
            return

        self.announce(
            "Initializing git submodule external/HillVallEA required for hillvallimpl...",
            level=2,
        )
        try:
            subprocess.check_call(
                [
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "external/HillVallEA",
                ],
                cwd=ROOT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Git is required to initialize external/HillVallEA but was "
                "not found on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            self.announce(
                "Submodule update failed. Falling back to cloning "
                "external/HillVallEA from .gitmodules.",
                level=2,
            )
            self._clone_hillvallea_from_gitmodules(exc)

        if not HILLVALLEA_SRC.exists() or not any(HILLVALLEA_SRC.glob("*.cpp")):
            raise RuntimeError(
                "HillVallEA source files were not found after submodule "
                "initialization."
            )

    def _clone_hillvallea_from_gitmodules(
        self, source_error: Exception
    ) -> None:
        gitmodules_file = ROOT / ".gitmodules"
        if not gitmodules_file.exists():
            raise RuntimeError(
                "Failed to initialize external/HillVallEA and .gitmodules "
                "is missing."
            ) from source_error

        try:
            repo_url = subprocess.check_output(
                [
                    "git",
                    "config",
                    "--file",
                    str(gitmodules_file),
                    "--get",
                    "submodule.external/HillVallEA.url",
                ],
                cwd=ROOT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to read submodule URL for external/HillVallEA from "
                ".gitmodules."
            ) from exc

        clone_target = ROOT / "external" / "HillVallEA"
        clone_target.parent.mkdir(parents=True, exist_ok=True)
        if clone_target.exists() and any(clone_target.iterdir()):
            return

        try:
            subprocess.check_call(
                ["git", "clone", repo_url, str(clone_target)], cwd=ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to clone external/HillVallEA from URL declared in "
                ".gitmodules."
            ) from exc

    def _refresh_hillvall_extension_sources(self) -> None:
        """Replace the placeholder source list with the real one now that
        the submodule is available."""
        sources = _hillvall_sources()
        for ext in self.extensions:
            if ext.name == "hillvallimpl":
                ext.sources = sources


# ---------------------------------------------------------------------------
# Extension list – only populated when the user opts in.
# ---------------------------------------------------------------------------
if BUILD_HILLVALL:
    # We import pybind11 here (not at module level) so that a pure-Python
    # install never fails due to a missing pybind11.
    import pybind11

    ext_modules = [
        Extension(
            name="hillvallimpl",
            # Use the bindings file as a placeholder; BuildExtWithSubmodules
            # will replace this list with the full set once the submodule is
            # cloned.
            sources=[_relative_to_root(HILLVALLEA_BINDINGS_SRC)],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                ".",
            ],
            language="c++",
            extra_compile_args=_cxx_compile_args(),
        )
    ]
else:
    ext_modules = []

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithSubmodules},
)
