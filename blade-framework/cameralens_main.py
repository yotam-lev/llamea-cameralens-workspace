"""
Single entrypoint for all CameraLens experiment runs.

Usage
-----
Interactive menu::

    python cameralens_main.py

Direct run (for scripts / SLURM)::

    python cameralens_main.py --run lens_mvp
    python cameralens_main.py --run lens_mvp,lens_v2

List available runs and exit::

    python cameralens_main.py --list
"""
import argparse
import importlib
import os
import sys
import types
from pathlib import Path

# ── One-time JAX setup ────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)

# ── One-time platform / LLM setup ────────────────────────────────
from config import get_llm, get_n_jobs, get_platform

# ── Run discovery ─────────────────────────────────────────────────

_RUNS_DIR = Path(__file__).parent / "camera_problem_runs"


def _discover_runs() -> dict[str, types.ModuleType]:
    """
    Scan ``camera_problem_runs/`` for Python files that expose both
    ``RUN_META`` and ``configure_run``.

    Returns:
        Ordered dict mapping stem name (e.g. ``"lens_mvp"``) to the
        imported module.
    """
    runs: dict[str, types.ModuleType] = {}
    for path in sorted(_RUNS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module_name = f"camera_problem_runs.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            print(f"[main] ⚠️  Could not import {module_name}: {exc}", file=sys.stderr)
            continue
        if hasattr(mod, "RUN_META") and hasattr(mod, "configure_run"):
            runs[path.stem] = mod
    return runs


def _print_run_list(runs: dict[str, types.ModuleType]) -> None:
    print("\nAvailable runs:")
    for i, (stem, mod) in enumerate(runs.items(), start=1):
        meta = mod.RUN_META
        print(f"  {i}. {stem:30s}  {meta.get('description', '')}")
    print()


def _interactive_select(runs: dict[str, types.ModuleType]) -> list[str]:
    """Prompt the user to select one or more runs interactively."""
    _print_run_list(runs)
    keys = list(runs.keys())
    while True:
        raw = input(
            "Enter run number(s) or name(s) separated by commas "
            "(or 'q' to quit): "
        ).strip()
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        selected: list[str] = []
        valid = True
        for token in raw.split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(keys):
                    selected.append(keys[idx])
                else:
                    print(f"  Invalid number: {token!r}", file=sys.stderr)
                    valid = False
                    break
            elif token in runs:
                selected.append(token)
            else:
                print(f"  Unknown run: {token!r}", file=sys.stderr)
                valid = False
                break
        if valid and selected:
            return selected
        print("  Please enter valid run numbers or names.\n", file=sys.stderr)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="CameraLens experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        metavar="NAME[,NAME…]",
        help="Comma-separated list of run names to execute (skips interactive menu).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available runs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure camera_problem_runs is importable (blade-framework root on path)
    framework_root = str(Path(__file__).parent)
    if framework_root not in sys.path:
        sys.path.insert(0, framework_root)

    runs = _discover_runs()
    if not runs:
        print("[main] No run configurations found in camera_problem_runs/.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        _print_run_list(runs)
        sys.exit(0)

    # ── Determine which runs to execute ──────────────────────────
    if args.run:
        selected_names: list[str] = []
        for token in args.run.split(","):
            token = token.strip()
            if token not in runs:
                print(f"[main] Unknown run: {token!r}", file=sys.stderr)
                print(f"       Available: {', '.join(runs)}", file=sys.stderr)
                sys.exit(1)
            selected_names.append(token)
    else:
        selected_names = _interactive_select(runs)

    # ── One-time setup ────────────────────────────────────────────
    platform_name = get_platform()
    print(f"[main] Platform : {platform_name}")

    llm = get_llm()
    n_jobs = get_n_jobs()
    print(f"[main] n_jobs   : {n_jobs}")

    # ── Execute selected runs ─────────────────────────────────────
    for name in selected_names:
        mod = runs[name]
        meta = mod.RUN_META
        print(f"\n[main] ─── Starting run: {meta.get('name', name)} ───")
        experiment = mod.configure_run(llm, n_jobs)
        experiment()
        print(f"[main] ─── Finished run: {meta.get('name', name)} ───\n")


if __name__ == "__main__":
    main()
