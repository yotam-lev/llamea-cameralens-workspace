# Agent Force Configuration: Scaled Operations

## 1. DataExtractorAgent (Upgraded)
- **Role:** File Parser & Graph Builder.
- **Task:** Extract JSON data, identify the global best fitness path, and construct a Mermaid.js string connecting `parent_id` to `id`.
- **Output:** `./extracted_gen_data/`, `fitness_history.csv`, and a `lineage.mmd` file.

## 2. PseudocodeGeneratorAgent (Concurrent)
- **Role:** Parallel Logic Distiller.
- **Directive:** Map optimization step methods to standard mathematical notation using LaTeX. Execute API calls asynchronously across a thread pool sized to the host machine's capabilities.

## 3. EvolutionaryComparatorAgent (Diff-Focused)
- **Role:** Micro-Analyzer.
- **Input:** A `difflib` output between parent and child code.
- **Directive:** Output a concise JSON or structured markdown block highlighting exactly *what* changed (e.g., "Learning rate schedule shifted from constant to exponential decay"). Calculate $\Delta f = f_{child} - f_{parent}$.

## 4. MacroAnalyzerAgent (NEW)
- **Role:** Epoch Summarizer.
- **Task:** Group generations into epochs (e.g., Gens 0-20, 20-50). Read the micro-analysis reports and summarize the global optimization strategy. 
- **Goal:** Identify macro-trends, such as when the algorithm shifts from global exploration (large bounds) to local exploitation (gradient-based micro-adjustments on specific lens curvatures).