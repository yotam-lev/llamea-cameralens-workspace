# LLaMEA Analysis Pipeline: Architectural Review & Improvements

## 1. Flaws in the Current Script (`analyze_run_v2.py`)
The existing script successfully extracts code but suffers from several critical bottlenecks that hinder scalability and accuracy for LLaMEA lens optimization parsing:

* **Monolithic Prompts & Token Bloat:** The `PseudocodeGeneratorAgent` feeds entire Python classes to the local LLM at once. For complex optimization classes, this exceeds the optimal context window, leading to hallucinated logic, skipped methods, and inconsistent mathematical notation.
* **Missing Caching (Redundant Compute):** Evolutionary algorithms often carry over unchanged methods from parent to child. The current script re-translates the entire class every generation, wasting compute on identical logic.
* **Hardcoded & Unsafe Pathing:** Output directories (`./Lens_v4_50000_F/`) are hardcoded. Running the script multiple times mixes outputs from different runs, creating unmanageable data swamps.
* **Ignored Conversation Logs:** The script ignores `conversationlog.jsonl`, missing crucial context about *why* the LLM made specific evolutionary leaps (cost, tokens, and raw dialogue).
* **Concurrency Overload:** The ThreadPoolExecutor fires requests concurrently to a local Ollama instance. Local LLMs typically process requests sequentially; parallel requests just queue up and can cause timeout errors or memory limits.

## 2. Proposed Architectural Improvements
Transitioning to a Jupyter Notebook architecture resolves these issues by allowing step-by-step verification and state management:

* **AST-Driven Method Chunking:** Instead of passing the whole class, use Python's `ast` module to parse the class and isolate individual methods (e.g., `__init__`, `LensDesignStep`, `mutate`). Feed these to the LLM one by one.
* **Semantic Dictionary Caching:** Implement a hash map for methods. Before querying the LLM, hash the raw Python method string (or its AST dump). If it exists in the dictionary, reuse the existing pseudocode. This drastically reduces LLM calls.
* **Dynamic Directory Management:** Route all outputs to `blade-framework/output-analysis/<Experiment_name>/`. Implement a pre-flight check that detects existing folders, prompts the user, and cleanly wipes old data to prevent duplication.
* **Dual-Log Merging:** Join `log.jsonl` (results and code) with `conversationlog.jsonl` (dialogue and tokens) using timestamps to create a unified dataset before extraction begins.

## 3. Validation Steps
To ensure the LLM is deterministically generating accurate pseudocode, implement the following validation routine in the final notebook cell:

1.  **Duplicate Ingestion:** Feed the system two identical optimization classes under different IDs (e.g., `Test_A` and `Test_B`).
2.  **Cache Verification:** Assert that `Test_B` triggers a 100% cache hit rate and makes exactly 0 calls to the LLM API.
3.  **Variable Integrity Check:** Regex search the resulting pseudocode for critical specific variables passed from the environment (e.g., ensuring `grad0_cont` is preserved in V4 logic, and `hess` is preserved when processing V5 logic).
4.  **Diff Assertions:** Run a standard programmatic diff on `pseudocode_Test_A.md` and `pseudocode_Test_B.md`. The diff must be exactly zero.