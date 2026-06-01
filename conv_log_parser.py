import os
import json
import re

def parse_generation_log_to_json(file_path):
    parsed_algorithms = []
    current_algo = None

    # These are the strings the framework uses to append new instructions 
    # after the feedback block. We use these to trim the feedback clean.
    prompt_signatures = [
        "Critically analyze", 
        "Design a more", 
        "Take a completely", 
        "Introduce a self", 
        "STRICT FORMATTING", 
        "Iterative Refinement"
    ]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = entry.get("content", "")
                role = entry.get("role", "")

                # 1. MODEL: Extract Code, Class Name, and Parent ID
                if role != "client":
                    # Extract the python code block
                    code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
                    # Extract class and parent
                    class_match = re.search(r"class\s+([A-Za-z0-9_]+)(?:\s*\(([A-Za-z0-9_]*)\))?\s*:", content)
                    
                    if code_match and class_match:
                        current_algo = {
                            "optimization_class": class_match.group(1),
                            "parent_id": class_match.group(2) if class_match.group(2) else "None",
                            "score": None,
                            "feedback": "",
                            "code": code_match.group(1).strip()
                        }
                
                # 2. CLIENT: Extract Feedback and Score for the recently generated code
                elif role == "client" and current_algo is not None:
                    if "Feedback:" in content:
                        # Grab everything after "Feedback:"
                        feedback_raw = content.split("Feedback:")[-1].strip()
                        
                        # Clean the trailing prompt instructions from the feedback
                        for sig in prompt_signatures:
                            if sig in feedback_raw:
                                feedback_raw = feedback_raw.split(sig)[0].strip()
                        
                        current_algo["feedback"] = feedback_raw
                        
                        # Extract the float/inf score from the feedback string
                        score_match = re.search(r"Mean loss:\s*([-\d\.inf]+)", feedback_raw, re.IGNORECASE)
                        if score_match:
                            current_algo["score"] = score_match.group(1)
                            
                    # Save the completed algorithm and reset
                    parsed_algorithms.append(current_algo)
                    current_algo = None

        return parsed_algorithms

    except Exception as e:
        print(f"\n[Error] Failed to read file: {e}")
        return None

def main():
    print("--- Evolutionary Code Extractor to JSON (V3) ---")
    file_path = input("Enter the full local path to your JSON log file: ").strip()
    
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
    print("1: ONLY Correct (Valid) Optimization Classes + Code")
    print("2: ONLY '-inf' Optimization Classes + Feedback + Code")
    print("3: ALL Optimization Classes + Scores + Code")
    
    mode = input("Enter 1, 2, or 3: ").strip()

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)

    filtered_data = []
    
    for alg in algorithms:
        score = alg["score"]
        is_inf = score in ['-inf', 'inf', '-infinity', 'infinity']
        
        # Clone the dict to modify it safely for output
        row = dict(alg)
        
        if mode == '1':
            if score and not is_inf:
                row["feedback"] = "" # Keep output clean for successful runs
                filtered_data.append(row)
        elif mode == '2':
            if is_inf or (alg["feedback"] and not score):
                filtered_data.append(row)
        elif mode == '3':
            filtered_data.append(row)
            
    if not filtered_data:
        print("\n[Notice] No data matched your criteria.")
        return

    out_name = f"{name}_extracted_code.json"
    out_path = os.path.join(dir_name, out_name)

    try:
        with open(out_path, mode='w', encoding='utf-8') as outfile:
            # Dump list of dictionaries to a well-formatted JSON file
            json.dump(filtered_data, outfile, indent=4)

        print(f"\n[Success] Extracted {len(filtered_data)} fully-coded algorithms.")
        print(f"JSON File saved to: {out_path}")

    except Exception as e:
        print(f"\n[Error] Could not write to output file: {e}")

if __name__ == "__main__":
    main()