import os
import re
import json
import csv
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class SemanticMeaningRegistry:
    """
    A thread-safe central registry for mapping normalized code/pseudocode lines
    to unique integer IDs (code_line_meaning).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._meaning_to_id: Dict[str, int] = {}
        self._id_to_meaning: Dict[int, str] = {}
        self._next_id = 1

    def get_or_create_id(self, normalized_line: str) -> int:
        with self._lock:
            if normalized_line in self._meaning_to_id:
                return self._meaning_to_id[normalized_line]
            
            line_id = self._next_id
            self._next_id += 1
            self._meaning_to_id[normalized_line] = line_id
            self._id_to_meaning[line_id] = normalized_line
            return line_id

    def get_meaning(self, line_id: int) -> str:
        with self._lock:
            return self._id_to_meaning.get(line_id, "")

    def get_all_meanings(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._id_to_meaning)


class SemanticMapperAgent:
    """
    Agent responsible for processing a single class's standardized pseudocode file.
    It cleans code comments/blocks, normalizes syntax, abstracts variable references
    and maps each line to a global code_line_meaning ID.
    """
    def __init__(self, registry: SemanticMeaningRegistry):
        self.registry = registry

    def normalize_line(self, line: str) -> str:
        """
        Normalizes a pseudocode line to ignore variable names and minor formatting differences.
        """
        # Remove comments if any
        line = re.split(r'#|//', line)[0]
        line = line.strip()
        if not line:
            return ""

        # Lowercase everything to avoid Case mismatches
        line = line.lower()
        
        # Replace variable tokens [VAR_X] and variable names VAR_X with [VAR]
        line = re.sub(r'\[var_\d+\]', '[var]', line)
        line = re.sub(r'\bvar_\d+\b', '[var]', line)
        
        # Normalize multiple spaces to a single space
        line = re.sub(r'\s+', ' ', line)
        
        # Remove spaces around common operators
        line = re.sub(r'\s*([=\+\-\*/\[\]\(\),:])\s*', r'\1', line)
        
        return line

    def analyze_file(self, class_id: str, filepath: Path) -> List[Tuple[int, str, int]]:
        """
        Parses a file and returns a list of tuples: (line_number, raw_line, meaning_id)
        """
        results = []
        if not filepath.exists():
            return results

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        in_code_block = False
        raw_line_number = 0

        for line in lines:
            raw_line_number += 1
            stripped = line.strip()

            # Skip markdown code block markers
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            # We parse all lines in the file (standardized pseudocode is normally in a block,
            # but we parse all non-empty lines to capture the full logic).
            normalized = self.normalize_line(line)
            if not normalized:
                continue

            # Skip header lines like 'class optimizer' or pseudocode boundary markers
            if normalized in ["class optimizer", "end class", "/// :::pseudocode:::", "/// :::end_pseudocode:::"]:
                continue

            meaning_id = self.registry.get_or_create_id(normalized)
            results.append((raw_line_number, line.strip(), meaning_id))

        return results


class ClusterMatrixAgent:
    """
    Agent responsible for collecting all mapped lines across classes,
    clustering identical meanings, and generating 2D matrices (JSON and CSV formats).
    """
    def __init__(self, registry: SemanticMeaningRegistry):
        self.registry = registry

    def generate_outputs(self, all_class_results: Dict[str, List[Tuple[int, str, int]]], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gather all class names and all meaning IDs
        classes = sorted(list(all_class_results.keys()))
        all_meanings = self.registry.get_all_meanings()
        meaning_ids = sorted(list(all_meanings.keys()))

        # 2. Build detailed JSON structure
        # Structure: Matrix[class_id][meaning_id] -> {count: int, lines: List[int]}
        matrix_json = {
            "classes": classes,
            "meanings": {str(m_id): meaning for m_id, meaning in all_meanings.items()},
            "matrix": {}
        }

        for class_id in classes:
            matrix_json["matrix"][class_id] = {}
            results = all_class_results[class_id]
            
            for line_num, _, m_id in results:
                m_str = str(m_id)
                if m_str not in matrix_json["matrix"][class_id]:
                    matrix_json["matrix"][class_id][m_str] = {
                        "count": 0,
                        "lines": []
                    }
                matrix_json["matrix"][class_id][m_str]["count"] += 1
                matrix_json["matrix"][class_id][m_str]["lines"].append(line_num)

        json_path = output_dir / "semantic_cluster_matrix.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(matrix_json, f, indent=2)
        print(f"✅ Saved detailed JSON matrix to {json_path}")

        # 3. Build CSV output
        # Columns: Class_ID, Meaning_1, Meaning_2, ...
        csv_path = output_dir / "semantic_cluster_matrix.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ["Class_ID"] + [f"Meaning_{m_id}" for m_id in meaning_ids]
            writer.writerow(header)

            for class_id in classes:
                row = [class_id]
                class_matrix = matrix_json["matrix"][class_id]
                for m_id in meaning_ids:
                    cell = class_matrix.get(str(m_id), {})
                    count = cell.get("count", 0)
                    row.append(count)
                writer.writerow(row)
        print(f"✅ Saved 2D CSV matrix to {csv_path}")


class OrchestratorAgent:
    """
    The orchestrator agent that manages finding files, spinning up SemanticMapperAgents
    in a thread pool, aggregating results, and calling ClusterMatrixAgent.
    """
    def __init__(self, output_base: Path):
        self.output_base = Path(output_base)
        self.registry = SemanticMeaningRegistry()
        self.mapper = SemanticMapperAgent(self.registry)
        self.clusterer = ClusterMatrixAgent(self.registry)

    def run(self):
        pseudocode_dir = self.output_base / "pseudocode_classes"
        if not pseudocode_dir.exists():
            raise FileNotFoundError(f"Pseudocode directory does not exist: {pseudocode_dir}")

        # Locate all Iteration_Final.md files
        class_folders = [d for d in pseudocode_dir.iterdir() if d.is_dir() and d.name.startswith("class_")]
        
        tasks = []
        for folder in class_folders:
            class_id = folder.name.replace("class_", "")
            # Look for Iteration_Final.md or final_pseudocode.md
            final_file = folder / "Iteration_Final.md"
            if not final_file.exists():
                final_file = folder / "final_pseudocode.md"
            
            if final_file.exists():
                tasks.append((class_id, final_file))

        print(f"Found {len(tasks)} classes with final pseudocode files.")

        all_class_results = {}
        
        # Parallel execution of SemanticMapperAgents
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.mapper.analyze_file, class_id, filepath): class_id
                for class_id, filepath in tasks
            }

            for future in as_completed(futures):
                class_id = futures[future]
                try:
                    results = future.result()
                    all_class_results[class_id] = results
                except Exception as e:
                    print(f"❌ Error processing class {class_id}: {e}")

        # Pass findings to ClusterMatrixAgent
        analysis_dir = self.output_base / "analysis"
        self.clusterer.generate_outputs(all_class_results, analysis_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent semantic pseudocode line clustering")
    parser.add_argument("--output-base", type=str, default=None, help="Path to output analysis folder")
    args = parser.parse_args()

    # Determine output base
    if args.output_base:
        output_base_path = Path(args.output_base)
    else:
        # Auto-detect latest lens_v5 directory under output-analysis
        current_dir = Path(__file__).resolve().parent.parent
        possible_dirs = list(current_dir.glob("lens_v5_*"))
        if possible_dirs:
            # Sort by name/timestamp (most recent first)
            possible_dirs.sort(key=lambda p: p.name, reverse=True)
            output_base_path = possible_dirs[0]
        else:
            output_base_path = current_dir

    print(f"Using output base path: {output_base_path}")
    orchestrator = OrchestratorAgent(output_base_path)
    orchestrator.run()
