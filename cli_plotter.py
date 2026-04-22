import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_log_data(filepath):
    """Loads the jsonl log file into a pandas DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                
    df = pd.DataFrame(data)
    
    # --- FIX: Force columns to numeric types ---
    df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
    df['fitness'] = pd.to_numeric(df['fitness'], errors='coerce')
    
    # Drop any rows where fitness or generation couldn't be parsed (e.g., failed runs)
    df = df.dropna(subset=['generation', 'fitness'])
    
    return df

def plot_generational_improvement(df, save_dir=None):
    """Plots the fitness of algorithms across generations."""
    plt.figure(figsize=(10, 6))
    
    # Scatter all runs
    plt.scatter(df['generation'], df['fitness'], alpha=0.5, color='gray', label='Generated Algorithms')
    
    # Calculate best per generation and global best
    best_per_gen = df.groupby('generation')['fitness'].min().reset_index()
    best_per_gen = best_per_gen.sort_values('generation')
    best_per_gen['global_best'] = best_per_gen['fitness'].cummin()
    
    # Plot trajectories
    plt.plot(best_per_gen['generation'], best_per_gen['fitness'], 'b--o', alpha=0.7, label='Best in Generation')
    plt.plot(best_per_gen['generation'], best_per_gen['global_best'], 'r-s', linewidth=2, label='Global Incumbent')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Lower is better)')
    plt.title('Algorithm Fitness Improvement per Generation')
    plt.xticks(range(int(df['generation'].min()), int(df['generation'].max()) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'generational_improvement.png'), dpi=300)
        print(f"Saved improvement plot to {save_dir}/generational_improvement.png")
    plt.show()

def plot_phylogeny_tree(df, save_dir=None):
    """Plots a tree showing parent-child algorithm relationships."""
    G = nx.DiGraph()
    
    # Add nodes
    for _, row in df.iterrows():
        short_id = str(row['id'])[:6]
        # Label shows short ID and fitness
        label = f"{short_id}\n{row['fitness']:.4f}"
        G.add_node(row['id'], label=label, generation=row['generation'], fitness=row['fitness'])
        
    # Add edges
    for _, row in df.iterrows():
        parents = row.get('parent_ids', [])
        if isinstance(parents, list):
            for p in parents:
                if p in G.nodes: # Ensure parent is actually in our dataset
                    G.add_edge(p, row['id'])
                    
    plt.figure(figsize=(14, 8))
    
    # Create custom layout: X = generation, Y = spread evenly
    pos = {}
    gen_counts = {}
    for node, data in G.nodes(data=True):
        gen = data.get('generation', 0)
        count = gen_counts.get(gen, 0)
        pos[node] = (gen, count)
        gen_counts[gen] = count + 1
        
    # Center the Y positions for each generation
    for node, data in G.nodes(data=True):
        gen = data.get('generation', 0)
        total_in_gen = gen_counts[gen]
        pos[node] = (pos[node][0], pos[node][1] - (total_in_gen - 1) / 2.0)
        
    # Extract fitness for color mapping
    fitnesses = [data['fitness'] for _, data in G.nodes(data=True)]
    
    # Draw network
    # Using viridis colormap. If lower fitness is better, mapping will handle it via colorbar.
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1800, node_color=fitnesses, 
                                   cmap=plt.cm.viridis, alpha=0.9, edgecolors='black')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', width=1.5)
    
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='white')
    
    # Add colorbar
    cbar = plt.colorbar(nodes, pad=0.02)
    cbar.set_label('Fitness Score (Lower is better)')
    
    plt.title('Algorithm Phylogeny Tree (Evolution of Optimizers)')
    
    # Formatting X-axis to show generation ticks
    plt.xticks(range(int(df['generation'].min()), int(df['generation'].max()) + 1))
    plt.xlabel('Generation')
    plt.gca().get_yaxis().set_visible(False) # Hide Y axis as it's just for spacing
    plt.box(False) # Remove border
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'phylogeny_tree.png'), dpi=300)
        print(f"Saved tree plot to {save_dir}/phylogeny_tree.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot LLaMEA algorithm generation logs.")
    parser.add_argument("log_file", type=str, help="Path to the log.jsonl file")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Could not find file {args.log_file}")
        return

    print(f"Loading data from {args.log_file}...")
    df = load_log_data(args.log_file)
    
    if df.empty:
        print("Error: No data found in the provided log file.")
        return
        
    print(f"Loaded {len(df)} algorithms across {df['generation'].nunique()} generations.")
    
    save_dir = os.path.dirname(os.path.abspath(args.log_file))
    
    plot_generational_improvement(df, save_dir)
    plot_phylogeny_tree(df, save_dir)

if __name__ == "__main__":
    main()