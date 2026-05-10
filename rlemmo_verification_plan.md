# Verification Plan: RLEMMO Baseline Algorithm

## 🎯 Goal
Verify the operational correctness and baseline performance of the `RLEMMO` evolutionary algorithm on the Double-Gauss objective within the BLADE framework.

## 🧪 Verification Strategy

### 1. Functional Correctness
- [ ] **Operator Testing:** Isolate each adaptive action (A1-A5). Ensure they properly mutate and respect bounds `[-1, 1]`.
- [ ] **Categorical Constraint Handling:** Verify the objective function properly rounds the glass ID variables (dimensions 18:24) as required by the Double-Gauss physics model.
- [ ] **Adaptive Mechanism:** Validate that operator weights adapt based on successful fitness improvements.

### 2. Integration Testing
- [ ] **BLADE Compatibility:** Run `RLEMMO` through the standard `Experiment` runner to ensure it interacts correctly with `ExperimentLogger` and logs fitness at each step.
- [ ] **Convergence Check:** Confirm that `RLEMMO` achieves a non-trivial fitness reduction (e.g., improves over pure random search).

### 3. Benchmarking Run
- [ ] Execute `RLEMMO` against the same Double-Gauss problem instance as the latest LLaMEA experiments (`v4_overnight`).
- [ ] Collect total fitness evolution and compute the "success rate" of individual operators.

## 📝 Execution Steps
1. Create a script `test_rlemmo_baseline.py` to run `RLEMMO` in isolation on the `ContextualLensOptimisation` problem.
2. Monitor `log.jsonl` output for fitness convergence.
3. Use the output stats (operator weights) to confirm that the bandit-based operator selection is functioning.

## 📅 Timeline
* **Phase 1: Unit Verification** (Operator and Constraint testing)
* **Phase 2: Integration Verification** (Running via Experiment runner)
* **Phase 3: Final Baseline Benchmarking**
