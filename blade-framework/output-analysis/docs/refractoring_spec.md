# Refactoring Specification: Optimization Output Analysis

## 1. Architectural Objective
Refactor the existing analysis pipeline to achieve strict separation of concerns. 
* **Logic:** All parsing, recursion, translation, and file-handling logic must reside in `output_analysis/src/analysis.py`.
* **Execution:** `output_analysis/notebooks/analysis.ipynb` will orchestrate the pipeline. 

**Core Constraint:** Every cell in the Jupyter Notebook (after the initial setup in Cell 1) must be executable in isolation. Functions in `analysis.py` must manage state via disk I/O (reading from the specific output folders) rather than relying on notebook memory state.

## 2. Cell-by-Cell Requirements for `analysis.ipynb`

### Cell 1: Initialization
* **Purpose:** Set up the environment.
* **Actions:** * Import dependencies.
  * Append `../src` to `sys.path` to allow importing `analysis.py`.
  * Define global variables for the current `<experiment name>` to be passed into subsequent functions.

### Cell 2: Parsing Optimization Classes
* **Purpose:** Isolate optimization classes and class variables.
* **Input Path:** `blade-framework/output_analysis/<experiment name>/stripped_data/optimisation_<id>`
* **Implementation:** Call a parsing function from `analysis.py` that reads the stripped data files, extracts the necessary classes/variables, and returns or saves the structured data.

### Cell 3: Recursion Logic
* **Purpose:** Execute and organize the recursion logic.
* **Implementation:** Call the relevant recursion wrapper from `analysis.py`. This step must load its required data from disk if run independently from Cell 2.

### Cell 4: Iterative Pseudocode Simplification
* **Purpose:** Execute the iterative simplification step.
* **Output Path:** Save outputs to `blade-framework/output_analysis/<experiment name>/pseudocode_classes/class_<id>/iteration_i.md` (where `i` represents the iteration index).
* **Implementation:** Call the simplification engine from `analysis.py`. It should cleanly iterate and save files to the designated paths.

### Cell 5: Clean Pseudocode Stickers
* **Purpose:** Finalize the pseudocode by cleaning "stickers" or artifact markers.
* **Input/Output Path:** Read the final iteration from the `pseudocode_classes/class_<id>/` directory and save the cleaned output as `iteration_final.md` in the exact same location.
* **Implementation:** String manipulation and regex logic must be handled entirely in `analysis.py`.

### Cell 6: Translation Engine
* **Purpose:** Translate the finalized pseudocode.
* **Input Path:** Scan for all `blade-framework/output_analysis/<experiment name>/pseudocode_classes/class_<id>/iteration_final.md` files.
* **Output Path:** Save the translated outputs to `blade-framework/output_analysis/<experiment name>/pseudocode_classes/class_<id>/iteration_translated.md`.
* **Secondary Output:** Save any additional or overarching analysis files to the subfolder `blade-framework/output_analysis/<experiment name>/analysis/`.

## 3. Design Principles for `analysis.py`
* **Stateless Functions:** Functions should expect file paths or data payloads as arguments.
* **Path Management:** Use `pathlib` for robust, cross-platform path resolution relative to the `blade-framework` root.
* **Error Handling:** Include sensible fallbacks or warnings if a cell is run in isolation but the prerequisite files (e.g., `iteration_final.md`) do not yet exist on disk.