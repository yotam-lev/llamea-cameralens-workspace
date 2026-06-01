# Antigravity IDE: LLaMEA Analysis Pipeline

## Execution Workflow
1. **Trigger:** `!analyze_run [path_to_log.jsonl]`
2. **Action 1 (Parsing):** Trigger `DataExtractorAgent`. 
   - *Logic:* Strip JSONL keys and save distinct generation files.
3. **Action 2 (Translation):** Trigger `PseudocodeGeneratorAgent`.
   - *Logic:* Pipe code blocks into local LLM interface with system instructions from `Requirements.md`.
4. **Action 3 (Comparison):** Trigger `EvolutionaryComparatorAgent`.
   - *Logic:* Compute mathematical difference between generations and log to `Analysis_Summary.md`.

## User Interface Configuration
- Display `Evolution_Report` in a side-by-side Diff view.
- Link `id` to the `parent_id` in the UI to allow click-through navigation of the lineage.