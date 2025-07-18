#!/usr/bin/env python3
"""
Convert pickled genome data to JSON for web visualization
"""
import pickle
import json
import numpy as np
import argparse
import os

def convert_genome_to_json(data_prefix="simulation"):
    """Convert pickled genome and config to JSON"""
    data_dir = "data"
    
    # Load genome
    genome_path = os.path.join(data_dir, f"{data_prefix}_best_genome.pkl")
    config_path = os.path.join(data_dir, f"{data_prefix}_config.pkl")
    
    if not os.path.exists(genome_path):
        print(f"Genome file not found: {genome_path}")
        return None
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None
    
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Verify genome structure
    if 'hidden_state' not in genome:
        print("Warning: No hidden_state found in genome, using default zero state")
    
    # Convert numpy arrays to lists for JSON serialization
    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        return obj
    
    # Prepare data structure
    network_data = {
        'config': {
            'input_dim': config.input_dim,
            'neurons': config.neurons,
            'output_dim': config.output_dim,
            'n_rays': config.n_rays,
            'use_bias': config.use_bias
        },
        'genome': {
            'W': numpy_to_list(genome['W']),  # Recurrent weights (hidden -> hidden)
            'E': numpy_to_list(genome['E']),  # Input weights (input -> hidden)
            'D': numpy_to_list(genome['D']),  # Output weights (hidden -> output)
            'b': numpy_to_list(genome['b']) if isinstance(genome['b'], np.ndarray) else genome['b'],
            'hidden_state': numpy_to_list(genome.get('hidden_state', 
                [0.0] * config.neurons if hasattr(config, 'neurons') else [0.0] * 5))  # Current hidden state
        },
        'performance': {
            'fitness': float(genome['fitness']),
            'age': float(genome['age']),
            'food_consumed': float(genome['food_consumed'])
        }
    }
    
    # Save to JSON
    json_path = os.path.join(data_dir, f"{data_prefix}_network.json")
    with open(json_path, 'w') as f:
        json.dump(network_data, f, indent=2)
    
    print(f"Network data exported to: {json_path}")
    return json_path

def main():
    parser = argparse.ArgumentParser(description='Convert genome pickle to JSON')
    parser.add_argument('--data-prefix', default='simulation', 
                       help='Prefix for data files')
    args = parser.parse_args()
    
    convert_genome_to_json(args.data_prefix)

if __name__ == "__main__":
    main() 