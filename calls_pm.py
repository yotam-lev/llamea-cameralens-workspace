import json
import re
from datetime import datetime

def analyze_ai_calls():
    file_path = "results-documentation/1504_Results_V4_5^4/results/lens_v4_False_50000/run-LLaMEA_v4_Memetic-DoubleGauss_v4-0_20260414_151726/log.jsonl"
    
    # Regex to match timestamps like "Apr 14 21:50:47 2026"
    time_pattern = re.compile(r'([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})')
    
    parsed_timestamps = []
    total_ai_calls = 0
    calls_per_gen = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                total_ai_calls += 1
                
                # Track counts per generation
                gen = data.get('generation', 'Unknown')
                calls_per_gen[gen] = calls_per_gen.get(gen, 0) + 1
                
                feedback = data.get('feedback', '')
                
                # Collect any available timestamps from this evaluation
                matches = time_pattern.findall(feedback)
                if matches:
                    for m in matches:
                        parsed_timestamps.append(datetime.strptime(m, "%b %d %H:%M:%S %Y"))
                        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    if len(parsed_timestamps) < 2:
        print("Not enough timestamps found to calculate total duration.")
        return

    # Sort all found timestamps chronologically to find the tracking window
    parsed_timestamps.sort()
    start_time = parsed_timestamps[0]
    end_time = parsed_timestamps[-1]
    
    total_duration_minutes = (end_time - start_time).total_seconds() / 60.0
    
    if total_duration_minutes > 0:
        cpm = total_ai_calls / total_duration_minutes
    else:
        cpm = 0
        
    print("--- Generation Breakdown ---")
    for g, count in sorted(calls_per_gen.items(), key=lambda x: str(x[0])):
        print(f"Generation {g}: {count} AI calls")
        
    print("\n--- Summary ---")
    print(f"Total AI calls (total configurations evaluated): {total_ai_calls}")
    print(f"Total tracked window: {total_duration_minutes:.2f} minutes")
    print(f"Average time per AI call & Evaluation: {(total_duration_minutes * 60) / total_ai_calls:.2f} seconds")
    print(f"True average calls per minute: {cpm:.4f}")

if __name__ == "__main__":
    analyze_ai_calls()