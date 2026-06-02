# Antigravity IDE: Scaled LLaMEA Pipeline

## Execution Workflow: `!analyze_run_v2`
1. **Trigger:** `!analyze_run_v2 [path_to_log.jsonl]`
2. **Action 1:** `DataExtractorAgent` parses the 50-100 generations, generating the Mermaid DAG and tracking the critical path of the highest-fitness lineage.
3. **Action 2:** `PseudocodeGeneratorAgent` runs concurrently to generate all pseudocode translations.
4. **Action 3:** The orchestrator scripts pre-compute `difflib` comparisons between parent and child pairs.
5. **Action 4:** `EvolutionaryComparatorAgent` evaluates the diffs.
6. **Action 5:** `MacroAnalyzerAgent` reviews all comparisons and compiles a comprehensive chronological timeline of the optimization strategy.

## File Output Architecture
Compile the final results into a master `Evolution_Dashboard.md` featuring:
- The Mermaid.js Lineage Graph.
- A table of Top 5 Highest Fitness Mutations.
- The Epoch Summary from the MacroAnalyzerAgent.