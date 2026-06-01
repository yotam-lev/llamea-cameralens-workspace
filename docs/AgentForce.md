# Agent Force Configuration
## 1. DataExtractorAgent
- **Role:** File Parser
- **Task:** Read `@blade-framework/results/Lens_Local_Test_2/log.jsonl`.
- **Output:** Save individual JSON files to `./extracted_gen_data/gen_{N}_{id}.json`.

## 2. PseudocodeGeneratorAgent
- **Role:** Logic Distiller
- **Input:** `gen_{N}_{id}.json`
- **Directive:** Feed the `code` field into the local LLM using the rules from `Requirements.md`.
- **Output:** Create `pseudocode_{id}.md`.

## 3. EvolutionaryComparatorAgent
- **Role:** Analysis Engine
- **Task:** Compare `pseudocode_{parent_id}.md` with `pseudocode_{child_id}.md`.
- **Output:** Generate `Evolution_Report_{child_id}.md` identifying specific parameter shifts in the optimization class.