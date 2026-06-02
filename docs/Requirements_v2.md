# Requirements: LLaMEA Scaled Analysis Pipeline (50-100 Gens)

## 1. Concurrency & Performance
- The pipeline must use `concurrent.futures` or `asyncio` to parallelize queries to the local Ollama API. 
- API timeout limits must be increased, with a robust exponential backoff for failed requests.

## 2. Syntactic Pre-Diffing (Token Optimization)
- Do not feed entire parent and child pseudocode blocks to the LLM for comparison. 
- Compute a strict syntactic diff (using Python's `difflib`) between the parent and child code. 
- Only send the isolated diffs (additions/subtractions) to the `EvolutionaryComparatorAgent` to evaluate the semantic and mathematical impact.

## 3. Visualization & Lineage
- **DAG Generation:** Output the lineage tree using Mermaid.js syntax (`graph TD`) to render visually in markdown.
- **Fitness Curve:** Generate a lightweight CSV of `generation,id,fitness` and optionally render a basic ASCII or Matplotlib plot of the global best fitness over time.

## 4. Mathematical Rigor
- Track the mathematical evolution of the perturbation logic. Specifically, flag changes in mutation distributions (e.g., transitioning from $\mathcal{U}(-1, 1)$ to $\mathcal{N}(0, \sigma^2)$) and step-size decay mechanisms.