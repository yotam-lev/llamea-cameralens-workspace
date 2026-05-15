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
*   **Performance insight:** was not acheiving very great results and would often achieve no evaluated results, the feww runs that did recieve a value received a value betwen 2 and 0.07 which is not good when comparing to state of the art results of 0.0005...

### 3. V4 (The Gradient-Aware Memetic Shift)
*   **Target:** `lens_v4.py`, `lens_v4_overnight_5000.py`
*   **Focus:** Bridging LLM optimization and physical gradient descent, provided grad0_cont variable which is the gradient of all continuous variables (provided from the camera-lens-simulation codebase).
*   **Performance Insight:** Discovered bottleneck in environment lifecycle (`_ensure_env` per evaluation), now optimized with persistent caching, environment was reinitiailising jax for every run instead of keeping an initialised env for all evaluations. Mean loss function improved with the 500 budget factor runs acheiving scores of 0.01 and the 50,000 budget factor score achieving 0.004. which is 8 times the state of the art result we are chasing, closer but still nowhere near.

*   ***Insights from all runs:***
- it was necessary to update to the qwen3.0:32b model as the 14b model was struggling to generate syntactically correct algorithms.
- it was found that generally elitism off generates greater results as it enables greater variation in parent nodes. elitism == true also would occaisionally cause a perpetual cycle of -inf results and in general scores -inf with far greater frequency then when it is turned off. 
- The most recent update to lens v4 saw the implementation of a more open prompt, this enables greater exploration for the llm which is backed up by the tsne plots generated. These have been experimented with budget factors of 500-1000 however not more as there is a persistant bug which I am finding difficult to resolve, causing evaluation to take far longer than expected. 

### 4. RLEMMO algorithm implementation

Run 1: The Single-Core Baseline
The Setup: This was your original run_rlemmo_lens.py script. It used a simple Multi-Armed Bandit (MAB) algorithm to randomly select exploration strategies. It ran on a single CPU core with a population of 100 explorers and a budget of 100,000 evaluations.

The Result: -0.0264

The Takeaway: This run proved the code worked but highlighted the flaw of standard evolutionary algorithms: getting stuck. Because the MAB algorithm is "blind" to its exact location on the map, it got trapped in a shallow local minimum and couldn't randomly guess its way out before the budget expired.

Run 2: The HPC Brute-Force (MAB Scaled)
The Setup: To utilize your cluster's 10 cores, we wrapped the baseline script in an iohblade.Experiment set to spawn 10 completely independent universes (runs=10) simultaneously. We also scaled the population up to 1,000 explorers to cast a wider net.

The Result: * Worst Run: -0.0287

Average (Median): -0.0156

Best Run: -0.0104

The Takeaway: A massive success. By giving the algorithm 10 independent chances, one of the search parties managed to bypass the local traps and find a valley that was more than twice as optimized as your initial baseline (-0.0104 vs -0.0264). However, the massive variance in the boxplot proved that MAB is highly volatile and relies heavily on "getting lucky."

Run 3: The Deep RL Local Prototype
The Setup: We completely tore down the MAB statistical array and replaced it with a context-aware PyTorch Neural Network (PPO). To test the complex new architecture safely, you ran a highly restricted local test (small population, tiny budget of ~1,000 steps).

The Result: -0.3427 and -0.4565 (Followed by a Monitor logger crash).

The Takeaway: As expected for a 1,000-step test, the actual lens fitness was quite poor (far away from zero). However, this run was a critical architectural victory. It proved the ray-tracer environments successfully converted to a Gym MDP, the neural network initialized correctly, and the Deep RL backpropagation worked. We fixed the logger crash immediately after.

Run 4: The Deep RL HPC Production Run
The Setup: With the code verified and the NVIDIA CUDA driver mismatch bypassed by forcing CPU gradient updates, you deployed the Deep RL agent to the cluster for a massive 800,000+ step run.

The Result: Plateaued strictly around -0.0107 (specifically tracking from -0.010718 down to -0.010709).

The Takeaway: The PyTorch AI successfully learned the macro-landscape! It swiftly navigated down to -0.0107, which is mathematically neck-and-neck with the absolute luckiest brute-force run (-0.0104). However, it suffered from "Entropy Collapse." Because we didn't give it microscopic step sizes or high enough rewards for tiny improvements, the neural network decided it was "good enough" and stopped exploring, plateauing at the 5th decimal place.



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
