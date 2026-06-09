import os
import json
import shutil
import logging
import re
import shutil
import argparse 
import textwrap
import urllib.request
import urllib.error
import hashlib
import time
from pathlib import Path
# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:14b"
FALLBACK_MODEL = "mistral:latest"
translation_cache = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def get_experiment_dir(experiment_name):
    """Return the output analysis directory for the given experiment name."""
    workspace = WORKSPACE_ROOT
    return workspace / "blade-framework" / "output_analysis" / experiment_name

def setup_directories(experiment_name, input_dir_str, overwrite=False):
    """Create the base directories for analysis."""
    output_base = get_experiment_dir(experiment_name)
    subfolders = ["stripped_data", "pseudocode_classes", "analysis"]
    
    if output_base.exists() and overwrite:
        shutil.rmtree(output_base)
        print(f"Deleted existing directory: {output_base}")
    elif output_base.exists():
        print(f"Directory {output_base} already exists. Using it.")
        
    for folder in subfolders:
        (output_base / folder).mkdir(parents=True, exist_ok=True)
    return output_base
# --- Parsing Logic ---

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
def parse_log_jsonl(log_path: Path, mapping: dict, offset: int = 0) -> int:
    """Parse log.jsonl and extract python code."""
    if not log_path.exists():
        logging.warning(f"File not found: {log_path}")
        return offset
    count = offset
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'id' in data and 'code' in data:
                    opt_id = f"optimisation_{count}"
                    mapping[opt_id] = {"uuid": data['id'], "source": "log.jsonl", "generation": data.get('generation')}
                    
                    code = clean_python_code(data['code'])
                    output_file = STRIPPED_DATA_DIR / f"{opt_id}.py"
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(code)
                    
                    count += 1
            except json.JSONDecodeError:
                logging.error(f"Failed to parse line in {log_path}")
                
    logging.info(f"Extracted {count - offset} entries from {log_path.name}")
    return count
def parse_conversationlog_jsonl(log_path: Path, mapping: dict, offset: int = 0) -> int:
    """Parse conversationlog.jsonl and extract python code from assistant responses."""
    if not log_path.exists():
        logging.warning(f"File not found: {log_path}")
        return offset
    count = offset
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Looking for model responses containing python code
                if data.get('role') != 'client' and 'content' in data:
                    content = data['content']
                    match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
                    if match:
                        code = match.group(1)
                        opt_id = f"optimisation_{count}"
                        mapping[opt_id] = {"uuid": None, "source": "conversationlog.jsonl"}
                        
                        code = clean_python_code(code)
                        output_file = STRIPPED_DATA_DIR / f"{opt_id}.py"
                        with open(output_file, 'w', encoding='utf-8') as out_f:
                            out_f.write(code)
                        
                        count += 1
            except json.JSONDecodeError:
                logging.error(f"Failed to parse line in {log_path}")
    logging.info(f"Extracted {count - offset} entries from {log_path.name}")
    return count

def load_jsonl(filepath):
    data = []
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"⚠️ Warning: File not found {filepath}")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
def parse_optimization_classes(experiment_name, input_dir_str):
    """
    Isolate optimization classes and class variables from log files.
    Input Path: blade-framework/output_analysis/<experiment name>/stripped_data/
    """
    output_base = get_experiment_dir(experiment_name)
    input_dir = Path(input_dir_str)
    
    stripped_dir = output_base / "stripped_data"
    stripped_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = input_dir / "log.jsonl"
    conv_log_path = input_dir / "conversationlog.jsonl"
    
    logs = load_jsonl(log_path)
    conv_logs = load_jsonl(conv_log_path)
    
    merged_data = {}
    for entry in logs:
        entry_id = entry.get("id")
        if entry_id:
            merged_data[entry_id] = {
                "id": entry_id,
                "generation": entry.get("generation", 0),
                "fitness": entry.get("fitness"),
                "parent": entry.get("parent_ids", []),
                "code": entry.get("code", ""),
                "feedback": entry.get("feedback", ""),
                "prompts": []
            }
            
    for conv in conv_logs:
        conv_id = conv.get("id") or conv.get("run_id")
        if conv_id and conv_id in merged_data:
            merged_data[conv_id]["prompts"].append(conv)
            
    variables_data = {}
    isolated_count = 0
    
    for entry_id, entry in merged_data.items():
        code_content = entry.get("code", "")
        if code_content.strip():
            py_filename = f"optimisation_{entry_id}.py"
            file_path = stripped_dir / py_filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            isolated_count += 1
            
            variables_data[entry_id] = {
                "id": entry_id,
                "generation": entry["generation"],
                "fitness": entry["fitness"],
                "parent": entry.get("parent", []),
                "feedback": entry["feedback"],
                "local_path": str(file_path.absolute())
            }
            
    variables_path = stripped_dir / "variables.json"
    with open(variables_path, "w", encoding="utf-8") as f:
        json.dump(variables_data, f, indent=4)
        
    print(f"📂 Isolated and saved {isolated_count} classes and variables.json to: {stripped_dir}")
    return variables_data
# --- LLM Utilities ---
def query_ollama(prompt, system_prompt=None, model=DEFAULT_MODEL, retries=3, delay=2):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                return res_json["message"]["content"]
        except Exception as e:
            print(f"⚠️ [Ollama Attempt {attempt} Error]: {e}")
            if attempt == retries:
                return query_ollama(prompt, system_prompt, model=FALLBACK_MODEL, retries=retries, delay=delay) if model == DEFAULT_MODEL else ""
            time.sleep(delay * (2 ** (attempt - 1)))
            
def get_chunk_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
def translate_chunk(chunk_text):
    chunk_hash = get_chunk_hash(chunk_text)
    if chunk_hash in translation_cache:
        return translation_cache[chunk_hash], True
        
    system_prompt = (
        "You are the PseudocodeGeneratorAgent.\n"
        "Your task is to rewrite a Python block into highly readable, standardized pseudocode.\n\n"
        "IMPORTANT: The Python code you receive may contain nested blocks that have ALREADY been translated into pseudocode. "
        "These translated blocks will be wrapped in `// :::PSEUDOCODE:::` and `// :::END_PSEUDOCODE:::` comments. "
        "You MUST seamlessly integrate these existing pseudocode blocks into your final translation of the surrounding Python code.\n\n"
        "RULES:\n"
        "1. Math Notation: Translate optimization math into LaTeX.\n"
        "2. Variable Preservation: Preserve exact variable names.\n"
        "3. Structural Mapping: Replace conditional constructs with uppercase statements (IF/THEN/ELSE, FOR, WHILE).\n"
        "Return ONLY the pseudocode block. Do not add conversational text."
    )
    
    prompt = f"Translate the following code block into pseudocode:\n\n```python\n{chunk_text}\n```"
    translated_pseudocode = query_ollama(prompt, system_prompt)
    translation_cache[chunk_hash] = translated_pseudocode
    return translated_pseudocode, False
# --- Simplification Engine ---
def is_empty_or_comment(line):
    s = line.strip()
    return not s or s.startswith('#') or s.startswith('//')
def get_indent(line):
    s = line.expandtabs(4)
    return len(s) - len(s.lstrip())
def single_pass_translate(lines):
    indents = []
    for i, line in enumerate(lines):
        if not is_empty_or_comment(line):
            indents.append((i, get_indent(line)))
            
    blocks = []
    for idx in range(len(indents) - 1):
        i, ind = indents[idx]
        next_i, next_ind = indents[idx+1]
        
        if next_ind > ind:
            end_idx = idx + 1
            has_nested = False
            while end_idx < len(indents) and indents[end_idx][1] > ind:
                if indents[end_idx][1] > next_ind:
                    has_nested = True
                end_idx += 1
                
            if not has_nested:
                start_line = i
                end_line = indents[end_idx-1][0]
                blocks.append((start_line, end_line))
                
    if not blocks:
        if not any(not is_empty_or_comment(l) for l in lines):
            return lines, False
            
        text = "\n".join(lines)
        pseudocode, _ = translate_chunk(text)
        pseudo_wrapped = ["// :::PSEUDOCODE:::"] + [f"// {l}" for l in pseudocode.split('\n')] + ["// :::END_PSEUDOCODE:::"]
        return pseudo_wrapped, True
        
    blocks.sort(key=lambda x: x[0], reverse=True)
    new_lines = list(lines)
    
    for start, end in blocks:
        block_lines = lines[start:end+1]
        block_text = "\n".join(block_lines)
        
        pseudocode, _ = translate_chunk(block_text)
        base_indent = get_indent(lines[start])
        prefix = " " * base_indent
        
        pseudo_wrapped = [f"{prefix}// :::PSEUDOCODE:::"]
        for p_line in pseudocode.split('\n'):
            pseudo_wrapped.append(f"{prefix}// {p_line}")
        pseudo_wrapped.append(f"{prefix}// :::END_PSEUDOCODE:::")
        
        new_lines[start:end+1] = pseudo_wrapped
        
    return new_lines, True
def run_iterative_simplification(experiment_name):
    """
    Execute iterative simplification step.
    Reads variables.json to find isolated classes, runs recursion, and outputs iteration_i.md
    """
    output_base = get_experiment_dir(experiment_name)
    variables_path = output_base / "stripped_data" / "variables.json"
    if not variables_path.exists():
        print(f"⚠️ Missing variables.json at {variables_path}. Run parse_optimization_classes first.")
        return
        
    with open(variables_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)
        
    pseudocode_dir = output_base / "pseudocode_classes"
    pseudocode_dir.mkdir(parents=True, exist_ok=True)
    
    for entry_id, entry in merged_data.items():
        py_filename = f"optimisation_{entry_id}.py"
        file_path = output_base / "stripped_data" / py_filename
        
        if not file_path.exists():
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        if not code_content.strip(): continue
        
        class_dir = pseudocode_dir / f"class_{entry_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        current_lines = code_content.split('\n')
        iteration = 1
        
        print(f"🚀 Starting Recursive Translation for Optimizer {entry_id[:8]}...")
        while True:
            new_lines, changed = single_pass_translate(current_lines)
            if not changed:
                break
                
            iter_path = class_dir / f"iteration_{iteration}.md"
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
                
            print(f"   ✅ Saved iteration {iteration} to {iter_path.name}")
            current_lines = new_lines
            iteration += 1
            
    print(f"✨ All recursive translations complete. Data saved to {pseudocode_dir}")
def extract_clean_markdown(lines):
    clean = []
    for line in lines:
        s = line.strip()
        if s in ["// :::PSEUDOCODE:::", "// :::END_PSEUDOCODE:::"]:
            continue
        if line.lstrip().startswith("// "):
            indent = len(line) - len(line.lstrip())
            clean.append(" " * indent + line.lstrip()[3:])
        else:
            clean.append(line)
    return clean
def clean_pseudocode_stickers(experiment_name):
    """
    Finalize pseudocode by cleaning stickers or artifact markers.
    Reads final iteration from pseudocode_classes/class_<id>/ and outputs iteration_final.md
    """
    output_base = get_experiment_dir(experiment_name)
    pseudocode_dir = output_base / "pseudocode_classes"
    if not pseudocode_dir.exists():
        print(f"⚠️ Missing pseudocode directory at {pseudocode_dir}.")
        return
        
    for class_dir in pseudocode_dir.iterdir():
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue
            
        iter_files = list(class_dir.glob("iteration_*.md"))
        iter_files = [f for f in iter_files if f.name != "iteration_final.md" and f.name != "iteration_translated.md"]
        if not iter_files: 
            continue
            
        latest_file = max(iter_files, key=lambda f: int(f.stem.split('_')[1]))
        
        with open(latest_file, "r", encoding="utf-8") as f:
            final_lines = f.read().split('\n')
            
        clean_lines = extract_clean_markdown(final_lines)
        
        final_md_path = class_dir / "iteration_final.md"
        with open(final_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_lines))
            
        print(f"✨ Cleaned {latest_file.name} -> {final_md_path.name} for {class_dir.name}")
        
    print("✨ Clean final pseudocode extracted successfully.")
def run_translation_engine(experiment_name):
    """
    Translate finalized pseudocode.
    Reads iteration_final.md and outputs iteration_translated.md.
    """
    output_base = get_experiment_dir(experiment_name)
    pseudocode_dir = output_base / "pseudocode_classes"
    if not pseudocode_dir.exists():
        print(f"⚠️ Missing pseudocode directory at {pseudocode_dir}.")
        return
        
    system_prompt = (
        "You are the TranslationEngineAgent.\n"
        "Your task is to review the provided pseudocode and standardize it into a highly formal, "
        "consistent mathematical pseudocode representation. Use LaTeX for math. Ensure exact variable preservation."
    )
        
    for class_dir in pseudocode_dir.iterdir():
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue
            
        final_md_path = class_dir / "iteration_final.md"
        if not final_md_path.exists():
            continue
            
        with open(final_md_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"🚀 Translating {class_dir.name}...")
        prompt = f"Standardize the following pseudocode:\n\n```\n{content}\n```"
        translated_pseudocode = query_ollama(prompt, system_prompt)
        
        translated_md_path = class_dir / "iteration_translated.md"
        with open(translated_md_path, "w", encoding="utf-8") as f:
            f.write(translated_pseudocode)
            
        print(f"   ✅ Saved to {translated_md_path.name}")
        
    print("✨ Translation engine completed successfully.")
