import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pathing ---
def find_workspace_root(current_dir, marker="blade-framework"):
    """Robustly find the workspace root containing the marker directory."""
    current = Path(current_dir).resolve()
    while current != current.parent:
        if (current / marker).is_dir():
            return current
        current = current.parent
    # Fallback to current working directory if not found
    return Path.cwd()

WORKSPACE_ROOT = find_workspace_root(__file__)
ANALYSIS_DIR = WORKSPACE_ROOT / "blade-framework" / "output-analysis"
OUTPUT_BASE = ANALYSIS_DIR / "output"

STRIPPED_DATA_DIR = OUTPUT_BASE / "stripped_data"
PSEUDOCODE_DIR = OUTPUT_BASE / "pseudocode_classes"
ANALYSIS_RESULTS_DIR = OUTPUT_BASE / "analysis"

# --- Extraction Functions ---
def clean_python_code(code: str) -> str:
    """Clean the extracted python code."""
    if "```python" in code:
        match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if match:
            code = match.group(1)
    elif "```" in code:
        match = re.search(r"```\n(.*?)\n```", code, re.DOTALL)
        if match:
            code = match.group(1)
    return code.strip() + "\n"

def clean_python_code_str(code: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()

def parse_log_jsonl(log_path: Path, mapping: dict, offset: int = 0) -> int:
    """Extract optimisation chunks from log.jsonl, ignoring errors and exact duplicates."""
    count = offset
    if not log_path.exists():
        logging.warning(f"File not found: {log_path}")
        return count
        
    seen_codes = set()
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if 'id' in data and 'code' in data:
                    if data.get('error'):
                        continue # Skip failed executions
                        
                    raw_code = data['code']
                    clean_code = clean_python_code_str(raw_code)
                    if clean_code in seen_codes:
                        continue
                        
                    seen_codes.add(clean_code)
                    opt_id = f"optimisation_{count}"
                    mapping[opt_id] = {
                        "uuid": data['id'],
                        "source": "log.jsonl"
                    }
                    output_file = STRIPPED_DATA_DIR / f"{opt_id}.py"
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(clean_code + "\n")
                    
                    count += 1
            except json.JSONDecodeError:
                continue
    logging.info(f"Extracted {count - offset} entries from log.jsonl")
    return count

def parse_conversationlog_jsonl(log_path: Path, mapping: dict, offset: int = 0) -> int:
    """Extract feedback prompts from conversationlog.jsonl where n % 2 == 0 (client lines), skipping n=0."""
    if not log_path.exists():
        logging.warning(f"File not found: {log_path}")
        return offset
        
    line_idx = 0
    feedback_idx = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if line_idx % 2 == 0 and line_idx > 0:
                    if data.get('role') == 'client' and 'content' in data:
                        opt_id = f"optimisation_{feedback_idx}"
                        if opt_id in mapping:
                            mapping[opt_id]["feedback_prompt"] = data['content']
                        feedback_idx += 1
            except json.JSONDecodeError:
                pass
            line_idx += 1
            
    logging.info(f"Extracted {feedback_idx} feedback prompts from conversationlog.jsonl")
    return offset

# --- Recursive Translation ---
def recursive_translation(mapping: dict):
    """Run deepest-first recursive translation using Ollama and Canonicalizer."""
    logging.info("Running recursive translation...")
    # Import modules dynamically or top-level. Here we do it inside to avoid circular imports if any, but since we are in the same dir it's fine.
    try:
        # pyrefly: ignore [missing-import]
        from src.canonicalizer import Canonicalizer
        # pyrefly: ignore [missing-import]
        from src.translator import Translator
    except ImportError:
        # Fallback to direct imports if run from inside src
        # pyrefly: ignore [missing-import]
        from canonicalizer import Canonicalizer
        # pyrefly: ignore [missing-import]
        from translator import Translator
        
    dict_path = OUTPUT_BASE / "translation_dict.json"
    
    for opt_id, info in mapping.items():
        input_file = STRIPPED_DATA_DIR / f"{opt_id}.py"
        output_file = PSEUDOCODE_DIR / f"{opt_id}_Iteration_Final.md"
        
        if not input_file.exists():
            continue
            
        canonicalizer = Canonicalizer(output_base=OUTPUT_BASE, class_source=opt_id, dict_path=str(dict_path))
        translator = Translator(canonicalizer, output_dir=PSEUDOCODE_DIR, opt_id=opt_id)
        
        logging.info(f"Translating {opt_id}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            code = f.read()
            
        final_pseudocode = translator.translate_code(code)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_pseudocode)
            
    logging.info("Recursive translation complete.")

# --- Similarity Metrics ---
def compute_similarity_metrics(mapping: dict):
    """Compute KPI similarity matrix using fuzzy/Levenshtein matching."""
    logging.info("Computing similarity metrics...")
    try:
        from src.similarity_metrics import analyze_pseudocode_similarities
    except ImportError:
        from similarity_metrics import analyze_pseudocode_similarities
        
    class_files = {}
    for opt_id in mapping.keys():
        filepath = PSEUDOCODE_DIR / f"{opt_id}_Iteration_Final.md"
        if filepath.exists():
            class_files[opt_id] = str(filepath)
            
    if not class_files:
        logging.warning("No pseudocode files found for similarity analysis.")
        return
        
    analyzer = analyze_pseudocode_similarities(class_files, threshold=85)
    
    # Save line ID mapping
    mapping_data = analyzer.get_line_id_mapping()
    mapping_path = ANALYSIS_RESULTS_DIR / "line_id_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4)
        
    # Get and save KPI sequence similarity matrix
    class_ids, sim_matrix = analyzer.get_sequence_similarity_matrix()
    matrix_path = ANALYSIS_RESULTS_DIR / "kpi_similarity_matrix.csv"
    
    with open(matrix_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("ClassID," + ",".join(class_ids) + "\n")
        # Write rows
        for i, row in enumerate(sim_matrix):
            row_strs = [f"{val:.4f}" for val in row]
            f.write(f"{class_ids[i]}," + ",".join(row_strs) + "\n")
            
    logging.info(f"Similarity metrics complete. Matrix saved to {matrix_path.name}")

# --- CLI Setup & Execution ---
def setup_directories(force=False):
    directories = [STRIPPED_DATA_DIR, PSEUDOCODE_DIR, ANALYSIS_RESULTS_DIR]
    
    for directory in directories:
        if directory.exists():
            if not force:
                response = input(f"Directory {directory.name} already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    logging.info(f"Skipping overwriting {directory.name}.")
                    continue
            logging.info(f"Clearing and recreating {directory.name}...")
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("Directories setup successfully.")

def main():
    parser = argparse.ArgumentParser(description="LLaMEA Output Analysis Pipeline v3")
    parser.add_argument("--force", action="store_true", help="Skip interactive prompts to overwrite directories")
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=str(WORKSPACE_ROOT / "blade-framework" / "results" / "Lens_v5_50000_False_03_06" / "llamea_run_lens_v5_50000_F_03_06"),
        help="Path to the directory containing log.jsonl and conversationlog.jsonl"
    )
    parser.add_argument(
    "--max-classes",
    type=int,
    default=0,
    help="Maximum number of classes to process (0 for unlimited)"
    )
    
    
    args = parser.parse_args()

    # 1. Setup Directories
    setup_directories(force=args.force)

    # 2. Paths
    log_dir = Path(args.log_dir).resolve()
    log_jsonl_path = log_dir / "log.jsonl"
    conversationlog_jsonl_path = log_dir / "conversationlog.jsonl"

    mapping = {}
    
    # 3. Parse files
    logging.info(f"Starting data extraction from {log_dir} ...")
    offset = parse_log_jsonl(log_jsonl_path, mapping, offset=0)
    # Only parse conversationlog if we haven't extracted everything from log.jsonl,
    # or just parse both sequentially to be safe
    parse_conversationlog_jsonl(conversationlog_jsonl_path, mapping, offset=offset)

    if not mapping:
        logging.warning("No optimisations were extracted. Please check the input logs.")
    
    if args.max_classes > 0:
        logging.info(f"Limiting to first {args.max_classes} optimisations.")
        mapping = dict(list(mapping.items())[:args.max_classes])


    # 4. Save mapping to variables.json
    variables_path = OUTPUT_BASE / "variables.json"
    with open(variables_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    logging.info(f"Saved mapping to {variables_path}")

    # 5. Execute further steps
    recursive_translation(mapping)
    compute_similarity_metrics(mapping)
    
    logging.info("Analysis pipeline v3 complete.")

if __name__ == '__main__':
    main()
