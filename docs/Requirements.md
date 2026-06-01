# Requirements: LLaMEA Optimization Analysis
## Objective
Trace the evolution of the `Optimizer` class across generations by converting raw `log.jsonl` into consistent, comparable pseudocode.

## Data Extraction Schema
Extract only the following fields:
- `id`: Unique identifier.
- `parent_ids`: To reconstruct the lineage tree.
- `fitness`: Score (Float).
- `feedback`: Performance summary.
- `code`: The source code to be parsed.

## Pseudocode Standardization Rules
- **Math Notation:** Use LaTeX (e.g., $L = \sum_{i} |y_i - \hat{y}_i|$) for loss functions.
- **Structural Mapping:** Replace `if/else` with `IF [condition] THEN...`, loops with `FOR [iterator]`.
- **Domain Mapping:** Rename `optimizer_step` to `LensDesignStep`.
- **Constraint:** Maintain original nested logic depth.