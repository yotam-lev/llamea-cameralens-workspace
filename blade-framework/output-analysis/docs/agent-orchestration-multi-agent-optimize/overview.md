# Multi-Agent Optimization & Analysis: Overview

This document provides a high-level overview of the multi-agent performance engineering framework tailored for the **LLaMEA Camera Lens Optimization** workspace. It outlines how specialized agents coordinate to analyze optimization logs, distill algorithm mechanics into canonical pseudocode, and evaluate their similarities to identify evolutionary trends.

## Core Objective

In LLM-driven evolutionary frameworks like **BLADE**, LLMs continuously generate new black-box optimization algorithms to minimize loss functions (e.g., the complex **Double-Gauss** lens loss function). Over several generations, the framework produces a large quantity of python scripts and conversation logs containing raw code, tracebacks, and prompts.

Understanding, profiling, and optimizing this pipeline is extremely challenging for a single monolithic system due to:
1. **Token Bloat / Context Window Limitations:** Passing complete optimization classes to an LLM results in hallucinated logic and high token cost.
2. **Compute Redundancy:** Evolved optimization steps carry forward unchanged helper functions from parents, wasting compute on redundant translations.
3. **Execution Latency:** Simulating physical ray tracing on double-gauss lenses is computationally expensive.
4. **Environment Discrepancies:** Slight shifts in library environments or wrapper bounds can silently degrade search quality.

By introducing a **Multi-Agent Performance Engineering** approach, we partition these concerns into dedicated agent personas that coordinate efficiently, optimize context windows, track latency bottlenecks, and preserve system integrity.

## Architecture & Integration

The multi-agent system integrates directly with the components of the workspace:

```mermaid
graph TD
    subgraph Physics Engine [Camera Lens Simulation]
        DG[DoubleGaussObjective]
    end

    subgraph Evolutionary Loop [BLADE Framework]
        LLaMEA[llamea.py / lens_v5.py]
        LO[lens_optimisation.py]
    end

    subgraph Output Analysis [Analysis Pipeline]
        AE[analysis_v4_1.py]
        C[canonicalizer.py]
        T[translator.py]
        SM[similarity_metrics.py]
    end

    subgraph Multi-Agent Swarm [Orchestration Toolkit]
        MAO[MultiAgentOrchestrator]
        CMA[CodebaseMappingAgent]
        COA[ContextOptimizerAgent]
        EPA[ExecutionProfilingAgent]
    end

    LLaMEA -->|Runs Experiment| LO
    LO -->|Evaluates Candidates| DG
    LO -->|Logs JSONL| Output
    Output -->|Extracted by| AE
    AE -->|Canonicalizes| C
    C -->|AST Chunking| T
    T -->|Calculates similarity| SM
    
    MAO -->|Coordinates| CMA
    MAO -->|Coordinates| COA
    MAO -->|Coordinates| EPA
    
    CMA -->|Maps Dependencies| Output Analysis
    COA -->|Manages Cache & Token Budget| T
    EPA -->|Profiles Memory & Latency| Physics Engine
```

## Performance & Optimization Strategies

Following the principles of [agent-orchestration-multi-agent-optimize](file:///Users/Yotam/Downloads/thesis_temporary_code/lens_v4_conitnuation/llamea-cameralens-workspace/.agent/skills/agent-orchestration-multi-agent-optimize/SKILL.md), the system employs several advanced performance strategies:

### 1. Context Window & Token Management
* **Indentation-Based AST Chunking:** The [Translator](file://../src/translator.py) parses python code into an AST and queries the local LLM only for isolated, nested blocks.
* **Semantic Caching:** The [Canonicalizer](file://../src/canonicalizer.py) hashes and normalizes code segments. If a segment's translation exists in `translation_dict.json`, the agent retrieves it from the cache instantly, bypassing the LLM.

### 2. Coordination Efficiency & Parallelism
* **Priority Queue-Based Workloads:** Tasks (such as translating, profiling, and plotting) are organized in a prioritized execution queue to minimize idle time.
* **Non-Blocking Thread Pool Execution:** Independent components (e.g. translating different optimization classes) are processed asynchronously to match local host computing cores.

### 3. Latency & Resource Monitoring
* **Physics Simulation Benchmarking:** Tracking ray-tracing simulation time under different optimization steps to flag solvers that exceed the `eval_timeout` threshold (e.g., complex CMA-ES steps).
* **LLM API Observability:** Recording the latency and token overhead of local Ollama API requests.

---

For specific details regarding agent personas and prompts, see [agents_specification.md](file://agents_specification.md).  
To review the structural maps of the codebase, see [codebase_map.md](file://codebase_map.md).
