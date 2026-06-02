import os
import json

def parse_generation_log_to_json(file_path):
    parsed_algorithms = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping line {line_num}: Invalid JSON")
                    continue

                # Extract fields directly from the structured JSON
                algo_id = entry.get("id", "Unknown")
                name = entry.get("name", "UnknownClass")
                parents = entry.get("parent_ids", [])
                parent_id = parents[0] if parents else "None"
                fitness = entry.get("fitness", None)
                feedback = entry.get("feedback", "")
                code = entry.get("code", "")
                description = entry.get("description", "")
                generation = entry.get("generation", 0)
                error = entry.get("error", "")

                # Consolidate error and feedback strings
                full_feedback = feedback
                if error and error not in feedback:
                    full_feedback += f" | Error: {error}"

                current_algo = {
                    "id": algo_id,
                    "generation": generation,
                    "optimization_class": name,
                    "parent_id": parent_id,
                    "score": str(fitness) if fitness is not None else None,
                    "description": description.strip(),
                    "feedback": full_feedback.strip(),
                    "code": code.strip()
                }
                
                parsed_algorithms.append(current_algo)

        return parsed_algorithms

    except Exception as e:
        print(f"\n[Error] Failed to read file: {e}")
        return None

def main():
    print("--- Evolutionary Code Extractor to JSON (log.jsonl Version) ---")
    file_path = input("Enter the full local path to your log.jsonl file: ").strip()
    
    if file_path.startswith(('"', "'")) and file_path.endswith(('"', "'")):
        file_path = file_path[1:-1]

    if not os.path.isfile(file_path):
        print("\n[Error] File not found. Check the path and try again.")
        return

    algorithms = parse_generation_log_to_json(file_path)
    if not algorithms:
        print("\n[Notice] No algorithms were found or the file was empty.")
        return

    print("\nWhat would you like to extract?")
    print("1: ONLY Correct (Valid) Optimization Classes (ignores -inf/inf)")
    print("2: ONLY '-inf' / Failed Optimization Classes")
    print("3: ALL Optimization Classes")
    
    mode = input("Enter 1, 2, or 3: ").strip()

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)

    filtered_data = []
    
    for alg in algorithms:
        score_str = str(alg["score"]).lower()
        is_inf = score_str in ['-inf', 'inf', '-infinity', 'infinity', 'none']
        
        row = dict(alg)
        
        if mode == '1':
            # Valid runs have a numeric score that is not inf
            if not is_inf:
                row["feedback"] = ""  # Keep output clean for successful runs
                filtered_data.append(row)
        elif mode == '2':
            # Failed runs have -inf score or explicit error feedback
            if is_inf or (row["feedback"] and not score_str):
                filtered_data.append(row)
        elif mode == '3':
            filtered_data.append(row)
            
    if not filtered_data:
        print("\n[Notice] No data matched your criteria.")
        return

    out_name = f"{name}_extracted_taxonomy_data.json"
    out_path = os.path.join(dir_name, out_name)

    try:
        with open(out_path, mode='w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, indent=4)

        print(f"\n[Success] Extracted {len(filtered_data)} algorithms.")
        print(f"JSON File saved to: {out_path}")

    except Exception as e:
        print(f"\n[Error] Could not write to output file: {e}")

if __name__ == "__main__":
    main()