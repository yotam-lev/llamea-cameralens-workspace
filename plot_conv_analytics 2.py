import os
import json
import argparse
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_conversation_log(filepath):
    """
    Parses the conversation log to extract timing, errors, and token data per iteration.
    An iteration is defined as: Client Prompt -> LLM Generation -> Client Feedback (Eval)
    """
    iterations = []
    current_iter = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            # Parse timestamp
            try:
                # Handle standard ISO format from Python datetime
                t = datetime.fromisoformat(row['time'])
            except ValueError:
                t = datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S.%f")

            role = row['role']
            content = row['content']
            
            if role == 'client':
                # If we already have a generated model response waiting for evaluation feedback
                if 'llm_time' in current_iter:
                    # Time between LLM generating code and Client giving feedback = Evaluation Time
                    current_iter['eval_time'] = (t - current_iter['llm_time']).total_seconds()
                    
                    # Extract error/feedback message
                    error_msg = "Success / Valid"
                    if "### Error Encountered" in content:
                        match = re.search(r'### Error Encountered\n(.*?)(?:\n|$)', content)
                        if match: error_msg = match.group(1).strip()
                    elif "Feedback:" in content:
                        match = re.search(r'Feedback:\n\n(.*?)(?:\n|$)', content)
                        if match and "Mean loss:" not in match.group(1): # Ignore success metrics
                            error_msg = match.group(1).strip()
                    
                    current_iter['error'] = error_msg
                    
                    # Save completed iteration
                    iterations.append(current_iter)
                    # Start a new iteration tracking from this client prompt
                    current_iter = {'client_time': t}
                else:
                    # Very first prompt
                    current_iter = {'client_time': t}
            else:
                # LLM response
                if 'client_time' in current_iter:
                    # Time between Client prompt and LLM response = Generation Time
                    current_iter['gen_time'] = (t - current_iter['client_time']).total_seconds()
                
                current_iter['llm_time'] = t
                current_iter['tokens'] = row.get('tokens', 0)
                
                # Extract code length
                code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
                current_iter['code_length'] = len(code_match.group(1)) if code_match else 0

    return pd.DataFrame(iterations)

def plot_analytics(df, save_dir):
    """Generates and saves three operational insight plots."""
    
    # Handle empty or incomplete data safely
    if df.empty:
        print("Not enough complete iterations to plot analytics.")
        return
        
    df['iteration'] = range(1, len(df) + 1)
    
    # Set global style
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # Plot 1: Timing Breakdown (Stacked Bar)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Plot Eval Time on bottom, Gen Time on top
    plt.bar(df['iteration'], df['eval_time'], label='Evaluation Time (Sandbox)', color='#e74c3c', alpha=0.8)
    plt.bar(df['iteration'], df['gen_time'], bottom=df['eval_time'], label='LLM Generation Time', color='#3498db', alpha=0.8)
    
    # Add a red dashed line indicating the 1800s timeout limit
    plt.axhline(y=1800, color='red', linestyle='--', alpha=0.6, label='1800s Timeout Threshold')
    
    plt.title('Time Consumption per Iteration: Generation vs. Evaluation', fontsize=14, pad=15)
    plt.xlabel('Algorithm Generation (Iteration)')
    plt.ylabel('Time Taken (Seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timing_breakdown.png'), dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: Error Frequency Distribution
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    
    # Clean up long error messages for the chart
    df['short_error'] = df['error'].apply(lambda x: x[:60] + '...' if len(x) > 60 else x)
    error_counts = df['short_error'].value_counts()
    
    # Color mapping: Green for success, shades of orange/red for errors
    colors = ['#2ecc71' if 'Success' in str(idx) else '#e67e22' for idx in error_counts.index]
    
    ax = error_counts.plot(kind='barh', color=colors)
    plt.title('Prominent Errors Encountered During Evaluation', fontsize=14, pad=15)
    plt.xlabel('Frequency')
    plt.ylabel('Error Message')
    ax.invert_yaxis()  # Highest frequency at the top
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Plot 3: LLM Code Length vs. Token Usage
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = '#2980b9'
    ax1.set_xlabel('Algorithm Generation (Iteration)')
    ax1.set_ylabel('Tokens Used', color=color)
    ax1.plot(df['iteration'], df['tokens'], marker='o', color=color, linewidth=2, label='Tokens')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = '#27ae60'
    ax2.set_ylabel('Generated Code Length (Characters)', color=color)
    ax2.plot(df['iteration'], df['code_length'], marker='s', color=color, linestyle='--', linewidth=2, label='Code Length')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('LLM Output Size Over Time', fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'output_size_trends.png'), dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot operational analytics from conversation logs.")
    parser.add_argument("log_file", type=str, help="Path to the conversationlog.jsonl file")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Could not find file {args.log_file}")
        return

    print(f"Parsing conversation data from {args.log_file}...")
    df = parse_conversation_log(args.log_file)
    
    if df.empty:
        print("Error: No complete iterations found in the provided log file.")
        return
        
    print(f"Successfully extracted {len(df)} iterations.")
    save_dir = os.path.dirname(os.path.abspath(args.log_file))
    
    plot_analytics(df, save_dir)
    print(f"\nAll plots have been saved to: {save_dir}")

if __name__ == "__main__":
    main()