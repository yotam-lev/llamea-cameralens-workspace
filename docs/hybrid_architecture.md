# Hybrid Compute Architecture: HPC-LLM / Mac-Eval

**Concept:**
To overcome HPC JIT-compilation overhead and thread deadlocks, this architecture decouples the LLaMEA algorithm generation from the evaluation of physical simulations.

**Components:**
1. **The Controller (Duranium HPC):** Hosts the LLaMEA framework and Ollama instance. It generates the `Optimizer` Python code strings.
2. **The Evaluator (M3 Pro Mac):** Hosts the `CameraLensEngine` simulator and a `ZeroMQ` evaluation server.
3. **The Link (SSH Tunnel):** A persistent reverse tunnel (`ssh -R`) mapping the Mac's port to the HPC.

**Communication Flow:**
1. LLaMEA generates `class Optimizer(...)`.
2. HPC sends the string to `tcp://localhost:5555`.
3. Mac receives the string, validates syntax, and evaluates it against the local JAX engine.
4. Mac returns the `loss` float to the HPC.