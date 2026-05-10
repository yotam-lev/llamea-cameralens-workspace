# Document Plan: Thesis on LLaMEA for Camera Lens Optimization

## 🎯 Thesis Objectives
This thesis investigates the application of Large Language Model-based Evolutionary Algorithms (LLaMEA) to the black-box optimization of complex camera lens systems (Double-Gauss model), using the `RLEMMO` algorithm as a performant baseline.

## 💡 Key R&D Challenges
*   **Prompt Engineering for Syntactic Correctness:** A significant portion of the development phase was dedicated to iterative prompt refinement. Ensuring the LLM generated syntactically correct, error-free Python code required strict API guardrails and extensive example-based prompting to prevent frequent failures.
*   **Environment Initialization & Dependency Management:** A major hurdle was ensuring that LLM-generated optimizers had consistent, reliable access to all required packages (`numpy`, `scipy`, `cma`, and local lens simulators). Managing these dependencies within an isolated execution context while maintaining performance was a critical challenge that shaped the current architecture.

## 📂 Experiment Evolution & Roadmap

### 1. MVP Phase (March 2026)
*   **Target:** `lens_mvp.py`, `2026-03-16_mvp_optimiser_code.py`
*   **Focus:** Initial integration of LLaMEA with the `DoubleGaussObjective`.

### 2. V2 & V3 Iterations (Late March - Early April)
*   **Target:** `lens_v2.py`, `lens_v3.py`
*   **Focus:** Introduction of prompt engineering to guide optimizer structure.

### 3. V4 (The Gradient-Aware Memetic Shift)
*   **Target:** `lens_v4.py`, `lens_v4_overnight_5000.py`
*   **Focus:** Bridging LLM optimization and physical gradient descent.
*   **Performance Insight:** Discovered bottleneck in environment lifecycle (`_ensure_env` per evaluation), now optimized with persistent caching.

### 4. RLEMMO Baseline Implementation
*   **Target:** `blade-framework/iohblade/methods/rlemmo.py`
*   **Purpose:** To serve as a high-quality, multimodal evolutionary algorithm baseline to validate LLaMEA performance against a state-of-the-art heuristic.
*   **Testing Focus:** Verification of multimodal capabilities, cluster-based search, and adaptive operator selection.

## 📝 Document Chapters

### Section 1: Introduction
*   Camera lens design as a high-dimensional, mixed-variable optimization challenge.
*   Motivation for LLM-driven vs. classical heuristic (RLEMMO) algorithm generation.

### Section 2: Background
*   Evolutionary Algorithms & Memetic Search.
*   LLaMEA Architecture.
*   Physics of Lens Design (Double-Gauss).

### Section 3: Approach
*   Framework design: The BLADE framework as a bridge between LLMs and black-box physics simulators.
*   Addressing LLM hallucination: Polymorphic wrappers, API guardrails, and environment stabilization.

### Section 4: Experiments
*   Evolution of the setup: From MVP (crashes, instability) to V4 (gradient-aware optimization, stable environment).
*   Baseline Comparison: Comparing LLaMEA performance against the RLEMMO baseline.
*   Metrics: Fitness score evolution, optimizer stability, computational budget usage.
*   Analysis of performance improvements: The impact of persistent virtual environments.

### Section 5: Conclusion
*   Summary of findings.
*   Future work: Scaling to larger objectives, improving gradient guidance in mixed-variable spaces.
