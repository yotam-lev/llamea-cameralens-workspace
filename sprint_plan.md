# Sprint Plan: Lens Optimization Investigation

## 🎯 Goal
Identify performance bottlenecks, specifically environment lifecycle overhead, and implement a persistent environment strategy to optimize experiment throughput.

## 📝 Task List

### 1. Performance Bottleneck Analysis
- [x] Profile `_ensure_env` in `iohblade/problem.py`.
- [x] Confirmed: Environment creation/reinstallation is the primary bottleneck.

### 2. Environment Lifecycle Optimization
- [ ] Refactor `iohblade/problem.py`: Implement environment caching to reuse the virtual environment for all evaluations within a run.
- [ ] Ensure package dependencies are checked/installed only once at start-up, rather than per-evaluation.
- [ ] Validate functional correctness to ensure that environment reuse does not cause state leakage between evaluations.

### 3. Logging & Prompt Verification
- [x] Verify logging persistence (`prompts.json`).
- [ ] Ensure that even with a persistent environment, the `run_eval.py` script continues to work correctly and safely.

### 4. Validation
- [ ] Execute a mini-benchmarking run with the persistent environment to measure speedup.
- [ ] Conduct a full validation run with `lens_v4_overnight_5000` using the new lifecycle.

## ⏳ Timeline
* **Phase 1: Environment Caching Implementation** (Current)
* **Phase 2: Functional Correctness Validation**
* **Phase 3: Final Experiment Run & Performance Benchmarking**
