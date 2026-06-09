import os
import re
import time
import json
import uuid
import keyword
from pathlib import Path

# Common python keywords, external libraries, and builtins to ignore as variables
IGNORE_WORDS = set(keyword.kwlist) | {
    "self", "np", "jnp", "math", "scipy", "cma", "jax",
    "len", "range", "int", "float", "list", "dict", "str", "bool",
    "print", "max", "min", "abs", "sum", "round", "zip", "enumerate",
    "append", "extend", "insert", "pop", "remove", "clear", "copy",
    "update", "get", "keys", "values", "items",
    "any", "all", "isinstance", "type",
    "def", "class", "return", "pass", "break", "continue",
    "shape", "dtype", "ndim", "size", "T", "inf", "nan", "float32", "float64"
}

class FileLock:
    """
    A simple process-safe and thread-safe file lock implementation
    using atomic directory or file creation flags (O_CREAT | O_EXCL).
    """
    def __init__(self, lock_path: Path, timeout: float = 10.0, delay: float = 0.05):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.delay = delay
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # os.O_CREAT | os.O_EXCL ensures atomic creation across processes/threads
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_path} within {self.timeout} seconds.")
                time.sleep(self.delay)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            os.close(self.fd)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

def save_variable_mappings(output_file: Path, mappings: list):
    """
    Saves new mappings to output_file using FileLock to ensure safety.
    """
    if not mappings:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = output_file.with_suffix(".lock")
    
    with FileLock(lock_file):
        data = []
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except Exception:
                data = []
        
        existing = {(item.get("original_var_name"), item.get("class_source")) for item in data}
        for m in mappings:
            key = (m.get("original_var_name"), m.get("class_source"))
            if key not in existing:
                data.append(m)
                existing.add(key)
                
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

class Canonicalizer:
    def __init__(self, output_base: Path, class_source: str, dict_path: str = None):
        self.output_base = Path(output_base)
        self.class_source = class_source
        self.dict_path = Path(dict_path) if dict_path else None
        self.var_map = {}  # original_var_name -> [VAR_X] token
        self.var_counter = 0
        self.output_file = self.output_base / "analysis" / "ind_variables.json"
        
        # Operation standardization mapping
        self.op_map = {
            r"(?i)np\.clip\(|jnp\.clip\(|CLIP\(": "[OP_BOUND](",
            r"(?i)\b([a-zA-Z_]\w*(?:\[[^\]]+\])?)\.astype\(int\)": r"[OP_TYPECAST](\1)",
            r"(?i)CAST\(([^,]+),\s*INTEGER\)": r"[OP_TYPECAST](\1)",
            r"(?i)\b([a-zA-Z_]\w*(?:\[[^\]]+\])?)\.copy\(\)": r"copy(\1)",
            r"(?i)(\[OP_BOUND\]\([^,]+),\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?\)": r"\1, INTEGER, INTEGER)"
        }
        
        self.translations = {}
        if self.dict_path and self.dict_path.exists():
            try:
                with open(self.dict_path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
            except Exception:
                pass

    def get_translation(self, raw_code: str) -> str:
        canon_code = self.canonicalize(raw_code, discover=True)
        return self.translations.get(canon_code)

    def add_translation(self, raw_code: str, translation: str):
        canon_code = self.canonicalize(raw_code, discover=True)
        self.translations[canon_code] = translation
        if self.dict_path:
            with open(self.dict_path, 'w', encoding='utf-8') as f:
                json.dump(self.translations, f, indent=2)

    def get_all_translations(self) -> dict:
        return self.translations

    def discover_variables(self, python_code: str):
        """
        Discovers actual variables from pure Python code block.
        Updates self.var_map and logs them to JSON state.
        Excludes words followed by '(' to avoid treating function names as variables.
        """
        # Exclude function and class names
        ignored_words = set(IGNORE_WORDS)
        for match in re.finditer(r"\b(?:def|class)\s+([a-zA-Z_]\w*)\b", python_code):
            ignored_words.add(match.group(1))

        # Split into lines
        lines = python_code.splitlines()
        defined_vars = set()
        mappings_to_log = []

        # Pre-scan for input variables (read before write)
        for idx, line in enumerate(lines):
            line_num = idx + 1
            
            # Detect loop variables
            match_for = re.match(r"\s*for\s+([^in]+)\s+in\s+", line)
            if match_for:
                for var in re.finditer(r"\b[a-zA-Z_]\w*\b", match_for.group(1)):
                    defined_vars.add(var.group(0))
            
            # Split line at the first assignment operator '=' not part of equality or comparison
            parts = re.split(r"(?<![=<>!])=(?![=])", line, maxsplit=1)
            if len(parts) == 2:
                lhs, rhs = parts[0], parts[1]
            else:
                lhs, rhs = "", parts[0]

            # Scan RHS for input variables (excluding function calls)
            for match in re.finditer(r"\b[a-zA-Z_]\w*\b(?!\s*\()", rhs):
                var_name = match.group(0)
                if var_name in ignored_words or var_name.startswith("OP_") or var_name.startswith("VAR_"):
                    continue
                if var_name not in defined_vars and var_name not in self.var_map:
                    token = f"[VAR_{self.var_counter}]"
                    self.var_counter += 1
                    self.var_map[var_name] = token
                    
                    mappings_to_log.append({
                        "id": str(uuid.uuid4()),
                        "token": token,
                        "original_var_name": var_name,
                        "class_source": self.class_source,
                        "line_id": f"{self.class_source}_line#{line_num}"
                    })

            # Scan LHS to mark variables as defined/mutated (excluding function calls)
            for match in re.finditer(r"\b[a-zA-Z_]\w*\b(?!\s*\()", lhs):
                var_name = match.group(0)
                if var_name in ignored_words or var_name.startswith("OP_") or var_name.startswith("VAR_"):
                    continue
                defined_vars.add(var_name)

        # Left-to-right scan for remaining undiscovered variables in the Python code
        for match in re.finditer(r"\b[a-zA-Z_]\w*\b(?!\s*\()", python_code):
            var_name = match.group(0)
            if var_name in ignored_words or var_name.startswith("OP_") or var_name.startswith("VAR_"):
                continue
            if var_name not in self.var_map:
                token = f"[VAR_{self.var_counter}]"
                self.var_counter += 1
                self.var_map[var_name] = token
                
                line_num = python_code[:match.start()].count("\n") + 1
                mappings_to_log.append({
                    "id": str(uuid.uuid4()),
                    "token": token,
                    "original_var_name": var_name,
                    "class_source": self.class_source,
                    "line_id": f"{self.class_source}_line#{line_num}"
                })

        if mappings_to_log:
            save_variable_mappings(self.output_file, mappings_to_log)

    def canonicalize(self, chunk: str, discover: bool = True) -> str:
        """
        Main entry point to perform tokenization, lineage tracking, and logical reduction on a code chunk.
        """
        # If discovery is requested (backward compatibility), we run discover_variables first
        if discover:
            # Clean Python
            clean_python = "\n".join(line for line in chunk.splitlines() if not line.strip().startswith("//") and not line.strip().startswith("#"))
            self.discover_variables(clean_python)

        # 1. Operation Standardization Mapping
        standardized_chunk = chunk
        for pattern, op_code in self.op_map.items():
            standardized_chunk = re.sub(pattern, op_code, standardized_chunk)

        # Replace all instances of original variable names with sequential tokens
        tokenized_chunk = standardized_chunk
        for var_name, token in sorted(self.var_map.items(), key=lambda x: len(x[0]), reverse=True):
            tokenized_chunk = re.sub(rf"\b{re.escape(var_name)}\b", token, tokenized_chunk)

        # 5. Logical Reduction Loop
        current_text = tokenized_chunk
        previous_text = ""
        while current_text != previous_text:
            previous_text = current_text
            current_text = self._reduce_step(current_text)

        return current_text

    def _reduce_step(self, text: str) -> str:
        """
        Executes a single step of copy propagation or dead variable pruning.
        Returns the modified text if a change was made, or the same text.
        """
        # A. Copy Propagation
        # Matches assignments like: [VAR_X] = [VAR_Y]
        match_cp = re.search(r"(?m)^(\s*)\[(VAR_\d+)\]\s*=\s*\[(VAR_\d+)\](?:\r?\n|$)", text)
        if match_cp:
            var_x = match_cp.group(2)
            var_y = match_cp.group(3)
            before = text[:match_cp.start()]
            after = text[match_cp.end():]
            # Replace [VAR_X] with [VAR_Y] only downstream (in 'after')
            after_replaced = re.sub(r"\[" + re.escape(var_x) + r"\]", f"[{var_y}]", after)
            return before + after_replaced

        # C. Single-use Attribute Unification
        # Matches [VAR_X] = self.[VAR_Y]
        match_sua = re.search(r"(?m)^(\s*)\[(VAR_\d+)\]\s*=\s*self\.\[(VAR_\d+)\](?:\r?\n|$)", text)
        if match_sua:
            var_x = match_sua.group(2)
            var_y = match_sua.group(3)
            # if VAR_Y is only used once (in this assignment)
            occurrences_y = len(re.findall(r"\[" + re.escape(var_y) + r"\]", text))
            if occurrences_y == 1 and var_x != var_y:
                # rename VAR_Y to VAR_X to unify the attribute name with the variable name
                before = text[:match_sua.start()]
                after = text[match_sua.end():]
                replaced_assignment = match_sua.group(0).replace(f"[{var_y}]", f"[{var_x}]")
                return before + replaced_assignment + after

        # B. Dead Variable Pruning
        # Matches any assignment to [VAR_Z]
        for match_dp in re.finditer(r"(?m)^(\s*)\[(VAR_\d+)\]\s*=.*?(?:\r?\n|$)", text):
            var_z = match_dp.group(2)
            # Count occurrences in the entire text
            occurrences = len(re.findall(r"\[" + re.escape(var_z) + r"\]", text))
            if occurrences == 1:
                # Remove this assignment line entirely
                before = text[:match_dp.start()]
                after = text[match_dp.end():]
                return before + after

        return text
