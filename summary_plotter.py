#!/usr/bin/env python3
"""
Standalone summary plotter that reads saved simulation data and shows plots
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import os
from typing import Any, Dict


def load_data(data_prefix="simulation"):
    """Load simulation data from files"""
    data_dir = "data"
    
    # Load CSV data
    csv_path = os.path.join(data_dir, f"{data_prefix}_history.csv")
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    
    # Load best genome
    pkl_path = os.path.join(data_dir, f"{data_prefix}_best_genome.pkl")
    best_genome = None
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            best_genome = pickle.load(f)
    
    # Load config
    cfg_path = os.path.join(data_dir, f"{data_prefix}_config.pkl")
    cfg = None
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
    
    return df, best_genome, cfg


def plot_summary(df, best_genome, cfg):
    """Create summary plots from loaded data"""
    if df is None:
        print("No data to plot")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Evolution Summary", fontsize=16)
    
    # Create subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main fitness plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['time'], df['best_fitness'], 'r-', label='Best Fitness', linewidth=2)
    ax1.plot(df['time'], df['avg_fitness'], 'b-', label='Avg Fitness', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Fitness')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title('Fitness Evolution')
    
    # Life and food plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['time'], df['avg_life'], 'm-', label='Avg Life', linewidth=1)
    ax2.plot(df['time'], df['best_life'], 'r-', label='Best Life', linewidth=2)
    ax2.plot(df['time'], df['avg_food'], 'c-', label='Avg Food', linewidth=1)
    ax2.plot(df['time'], df['best_food'], 'g-', label='Best Food', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Life / Food')
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_title('Life & Food Statistics')
    
    # Population plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['time'], df['population_size'], 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Population Size')
    ax3.grid(alpha=0.3)
    ax3.set_title('Population Dynamics')
    
    # Stats summary
    ax4 = fig.add_subplot(gs[1, 1])
    stats_text = f"""
Final Statistics:
• Best Fitness: {df['best_fitness'].iloc[-1]:.2f}
• Avg Fitness: {df['avg_fitness'].iloc[-1]:.2f}
• Final Population: {df['population_size'].iloc[-1]:.0f}
• Best Life: {df['best_life'].iloc[-1]:.1f}
• Best Food: {df['best_food'].iloc[-1]:.0f}
• Simulation Time: {df['time'].iloc[-1]:.1f}s
• Total Steps: {len(df)}
"""
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10, family='monospace')
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    # Neural network plot
    ax5 = fig.add_subplot(gs[:, 2])
    if best_genome and cfg:
        _draw_network(ax5, best_genome, cfg)
    else:
        ax5.text(0.5, 0.5, 'No genome data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.axis('off')
    
    plt.tight_layout()
    plt.show()


def _draw_network(ax, genome: Dict[str, Any], cfg):
    """Draw best agent neural network with weights"""
    ax.set_title(
        f"Best Agent Neural Network\n"
        f"Fitness: {genome['fitness']:.2f}, Life: {genome['age']:.1f}s, Food: {genome['food_consumed']:.0f}",
        fontsize=12
    )
    ax.axis('off')

    # Layout params
    input_n = cfg.input_dim
    hidden_n = cfg.neurons
    output_n = cfg.output_dim

    layer_x = [0, 3, 6]
    neuron_spacing = 0.3
    neuron_radius = 0.12

    # Compute positions
    def positions(n, x):
        if n == 1:
            return [(x, 0)]
        start = -(n - 1) * neuron_spacing / 2
        return [(x, start + i * neuron_spacing) for i in range(n)]

    input_pos = positions(input_n, layer_x[0])
    hidden_pos = positions(hidden_n, layer_x[1])
    output_pos = positions(output_n, layer_x[2])

    # Draw neurons
    for i, (x, y) in enumerate(input_pos):
        ax.add_patch(patches.Circle((x, y), neuron_radius, color='lightblue', ec='black', linewidth=1))
        ax.text(x, y, f'S{i}', ha='center', va='center', fontsize=8)
    
    for i, (x, y) in enumerate(hidden_pos):
        ax.add_patch(patches.Circle((x, y), neuron_radius, color='lightgreen', ec='black', linewidth=1))
        ax.text(x, y, f'H{i}', ha='center', va='center', fontsize=8)
    
    output_labels = ['Steer', 'Speed']
    for i, (x, y) in enumerate(output_pos):
        ax.add_patch(patches.Circle((x, y), neuron_radius, color='salmon', ec='black', linewidth=1))
        ax.text(x, y, output_labels[i], ha='center', va='center', fontsize=8)

    # Draw connections with normalized thickness
    E = genome['E']  # hidden x input
    D = genome['D']  # output x hidden
    max_w = max(np.abs(E).max(), np.abs(D).max(), 1e-6)

    def draw_connections(src_pos, dst_pos, weights):
        for j, (dx, dy) in enumerate(dst_pos):
            for i, (sx, sy) in enumerate(src_pos):
                w = weights[j, i]
                alpha = min(abs(w) / max_w, 1.0)
                if alpha < 0.15:  # Skip very weak connections
                    continue
                color = 'red' if w < 0 else 'blue'
                ax.plot([sx, dx], [sy, dy], color=color, alpha=alpha, linewidth=alpha * 3)

    # Draw input to hidden connections
    draw_connections(input_pos, hidden_pos, E)
    
    # Draw hidden to output connections
    draw_connections(hidden_pos, output_pos, D)

    # Set axis limits
    ax.set_xlim(-1, 7)
    ax.set_ylim(-max(input_n, hidden_n, output_n) * neuron_spacing / 2 - 1,
                max(input_n, hidden_n, output_n) * neuron_spacing / 2 + 1)
    
    # Add legend
    ax.text(0, -max(input_n, hidden_n, output_n) * neuron_spacing / 2 - 0.7,
            'Input\n(Sensors)', ha='center', va='center', fontsize=10)
    ax.text(3, -max(input_n, hidden_n, output_n) * neuron_spacing / 2 - 0.7,
            'Hidden\n(Memory)', ha='center', va='center', fontsize=10)
    ax.text(6, -max(input_n, hidden_n, output_n) * neuron_spacing / 2 - 0.7,
            'Output\n(Actions)', ha='center', va='center', fontsize=10)


def main():
    parser = argparse.ArgumentParser(description='Plot simulation summary from saved data')
    parser.add_argument('--data-prefix', default='simulation', 
                       help='Prefix for data files (default: simulation)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export network data to JSON for HTML visualizer')
    args = parser.parse_args()
    
    print(f"Loading data with prefix: {args.data_prefix}")
    df, best_genome, cfg = load_data(args.data_prefix)
    
    if df is None:
        print("Failed to load data. Make sure you have run the simulation with --save-data flag.")
        sys.exit(1)
    
    print(f"Loaded {len(df)} data points")
    plot_summary(df, best_genome, cfg)
    
    # Export JSON data for HTML visualizer
    if args.export_json:
        try:
            import json
            import numpy as np
            
            # Convert numpy arrays to lists for JSON serialization
            def numpy_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Prepare data structure
            network_data = {
                'config': {
                    'input_dim': cfg.input_dim,
                    'neurons': cfg.neurons,
                    'output_dim': cfg.output_dim,
                    'n_rays': cfg.n_rays,
                    'use_bias': cfg.use_bias
                },
                'genome': {
                    'W': numpy_to_list(best_genome['W']),  # Recurrent weights (hidden -> hidden)
                    'E': numpy_to_list(best_genome['E']),  # Input weights (input -> hidden)
                    'D': numpy_to_list(best_genome['D']),  # Output weights (hidden -> output)
                    'b': numpy_to_list(best_genome['b']) if isinstance(best_genome['b'], np.ndarray) else best_genome['b'],
                    'hidden_state': numpy_to_list(best_genome.get('hidden_state', 
                        [0.0] * cfg.neurons if hasattr(cfg, 'neurons') else [0.0] * 5))  # Final hidden state
                },
                'performance': {
                    'fitness': float(best_genome['fitness']),
                    'age': float(best_genome['age']),
                    'food_consumed': float(best_genome['food_consumed'])
                }
            }
            
            # Save to JSON
            json_path = os.path.join("data", f"{args.data_prefix}_network.json")
            with open(json_path, 'w') as f:
                json.dump(network_data, f, indent=2)
            
            print(f"\n✅ Network data exported to: {json_path}")
            print("\n💡 To view the interactive HTML visualization:")
            print("   1. Open network_visualizer.html in your browser")
            print(f"   2. Load the {json_path} file")
            
        except Exception as e:
            print(f"Warning: Could not export JSON data: {e}")


if __name__ == "__main__":
    main() 