import os
import json
import shutil
from datetime import datetime
from pathlib import Path


# --- Robust Path Navigation ---
while Path.cwd().name != 'llamea-cameralens-workspace' and Path.cwd().parent != Path.cwd():
    os.chdir('..')

if Path.cwd().name != 'llamea-cameralens-workspace':
    print(f"⚠️ Warning: Could not find 'llamea-cameralens-workspace'. Current dir: {Path.cwd()}")
else:
    print(f"✅ Successfully navigated to workspace root: {Path.cwd()}")

# --- Configuration ---
INPUT_DIR = Path("blade-framework/results/Lens_v5_50000_False_03_06/llamea_run_lens_v5_50000_F_03_06")
EXPERIMENT_NAME = "lens_v5_50000_False_v3"


current_date = datetime.now().strftime("%d-%m-%Y")
OUTPUT_BASE = Path(f"blade-framework/output-analysis/{EXPERIMENT_NAME}_{current_date}")
SUBFOLDERS = ["stripped_data", "pseudocode_classes", "analysis"]

if OUTPUT_BASE.exists():
    already_exists = True
else:
    already_exists = False
    for folder in SUBFOLDERS:
        (OUTPUT_BASE / folder).mkdir(parents=True, exist_ok=True)

print(f"✅ Setup complete. Output base: {OUTPUT_BASE}")


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def clean_and_normalize_source(code_str):
    import ast
    import re
    try:
        parsed = ast.parse(code_str)
        clean_code = ast.unparse(parsed)
        
        def replacer(m):
            if m.group(1): 
                return m.group(1)
            else:
                return '='
                
        # Regex matches strings (group 1) OR spaces around = (group 2, uncaptured here)
        regex = r'("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')|(?<![=<>!])\s*=\s*(?![=])'
        
        res = []
        for line in clean_code.splitlines():
            if not line.strip(): continue
            new_line = re.sub(regex, replacer, line)
            res.append(new_line)
        return "\n".join(res)
    except Exception:
        # Fallback if there's syntax errors
        return code_str

variables_path = OUTPUT_BASE / "stripped_data/variables.json"
if variables_path.exists():
    print(f"✅ Variables file already exists at {variables_path}.")
    response = input("Do you want to overwrite it? (y/n): ").strip().lower()
    if response != 'y':
        print("Exiting without overwriting.")
        exit(0)

log_path = INPUT_DIR / "log.jsonl"
conv_log_path = INPUT_DIR / "conversationlog.jsonl"

if log_path.exists():
    logs = load_jsonl(log_path)
else:
    raise FileNotFoundError(f"Log file not found: {log_path}")

if conv_log_path.exists():
    conv_logs = load_jsonl(conv_log_path)
else:
    raise FileNotFoundError(f"Conversation log file not found: {conv_log_path}")

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

print(f"Successfully mapped {len(merged_data)} optimization class entries.")

stripped_dir = OUTPUT_BASE / "stripped_data"
variables_data = {}
isolated_count = 0

for entry_id, entry in merged_data.items():
    code_content = entry.get("code", "")
    if code_content.strip():
        cleaned_code = clean_and_normalize_source(code_content)
        py_filename = f"optimisation_{entry_id}.py"
        file_path = stripped_dir / py_filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)
        isolated_count += 1
        
        variables_data[entry_id] = {
            "id": entry_id,
            "generation": entry["generation"],
            "fitness": entry["fitness"],
            "parent": entry.get("parent", []),
            "feedback": entry["feedback"],
            "code": cleaned_code,
            "local_path": str(file_path.absolute())
        }

variables_path = stripped_dir / "variables.json"
with open(variables_path, "w", encoding="utf-8") as f:
    json.dump(variables_data, f, indent=4)

print(f"📂 Isolated and saved {isolated_count} classes and variables.json to: {stripped_dir}")