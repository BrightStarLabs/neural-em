# 🧠 Neural Evolution — Emergent Agents Simulation

A sophisticated 2D simulation of autonomous agents with evolved neural network brains, featuring memory inheritance, proportional mutations, and interactive analysis tools.

Each agent:
- **Thinks** using a recurrent neural network with persistent memory
- **Senses** the world through dual-channel ray-casting (food and other agents)
- **Moves** based on learned motor commands (steering and speed)
- **Survives** by managing life energy and consuming food
- **Evolves** through reproduction with genetic and memory inheritance

Agents evolve over generations through natural selection, developing sophisticated behaviors for survival and reproduction.

---

## 🚀 Key Features

### **Neural Architecture**
- **Recurrent neural networks** with persistent memory states
- **Dual-channel sensors** (5 rays each for food and agents = 10 inputs)
- **Motor outputs** for steering and speed control
- **Evolutionary optimization** through reproduction and mutation

### **Memory System**
- **Persistent memory** across agent lifetime
- **Memory inheritance** from parent to child with configurable mutation
- **State visualization** in interactive HTML viewer

### **Advanced Genetics**
- **Proportional mutations** that scale with parameter values
- **Weight decay regularization** applied at birth
- **Configurable mutation rates** for all network components

### **Analysis & Visualization**
- **Real-time Pygame visualization** with sensor ray display
- **Comprehensive data logging** with CSV export
- **Interactive HTML network visualizer** with mini-simulation
- **Automatic plot generation** for fitness evolution and statistics

### **Technical Excellence**
- **Vectorized NumPy simulation** for high performance
- **Modular architecture** with clean separation of concerns
- **Toroidal world** with wrap-around physics
- **Configurable parameters** through YAML config files

---

## 🖥 Usage

### ▶ Basic Simulation
```bash
# Headless mode (terminal output)
python main.py

# Visual mode with Pygame GUI
python main.py --vis

# Set deterministic random seed
python main.py --seed 42
```

### ▶ Data Collection & Analysis
```bash
# Run simulation with data collection
python main.py --save-data --data-prefix my_experiment --vis

# This automatically:
# 1. Saves fitness history to CSV
# 2. Saves best agent genome to pickle
# 3. Generates summary plots
# 4. Exports JSON for HTML visualizer
```

### ▶ Manual Analysis
```bash
# Generate plots only
python summary_plotter.py --data-prefix my_experiment

# Export JSON for HTML visualizer
python summary_plotter.py --data-prefix my_experiment --export-json

# View exported network data
python export_network_data.py --data-prefix my_experiment
```

### ▶ Interactive Network Visualization
1. Run simulation with `--save-data`
2. Open `network_visualizer.html` in your browser
3. Load the generated JSON file from the `data/` folder

**Features:**
- **3D network layout** with input, hidden (reservoir), and output layers
- **Connection strength visualization** through transparency and thickness
- **Self-connections** rendered as loops
- **Interactive simulation** from current memory state
- **Weight matrix heatmaps** for all network parameters
- **Real-time state updates** during mini-simulation

---

## 🧬 Neural Architecture

### Agent Brain Structure
Each agent has a recurrent neural network with:

```
Input Layer (10 neurons)
    ↓
Hidden Layer (configurable, default 10)
    ↓
Output Layer (2 neurons: steer, speed)
```

### Mathematical Model
The neural network follows the update rule:

```
S[t+1] = tanh(W · S[t] + E · x[t] + b)
y[t] = D · S[t+1]
```

Where:
- `S[t]` = hidden state (memory) at time t (n-dimensional)
- `x[t]` = sensory input (10D: 5 food + 5 agent distances)  
- `y[t]` = motor output (2D: steer ∈ [-1,1], speed ∈ [0,1])
- `W` = recurrent weight matrix (n×n)
- `E` = input projection matrix (n×10)
- `D` = output decoder matrix (2×n)
- `b` = bias vector (n-dimensional)

### Memory Inheritance
During reproduction, children inherit parent's memory state:
```
S[0]_child = S[t]_parent + N(0, σ²_memory)
```

This allows evolved "instincts" to be passed between generations.

---

## 🔬 Evolutionary System

### Proportional Mutation
The system uses a sophisticated mutation scheme that scales with parameter values:

```
param_new = param_old × (1 + μ) × (1 - δ) + μ × k
```

Where:
- `μ ~ N(0, σ²_mutation)` = proportional mutation (Gaussian noise)
- `δ` = decay factor (weight regularization)
- `k` = small constant (0.01) for additional variation

### Reproduction Process
1. **Eligibility**: Agent must have life ≥ `reproduction_min_life`
2. **Probability**: Each eligible agent has `reproduction_prob` chance per step
3. **Cost**: Parent transfers `life_transfer` life to child
4. **Genetics**: Child inherits mutated genome with weight decay
5. **Memory**: Child inherits parent's memory state with mutation
6. **Population**: Limited by `max_agents` cap

### Fitness & Selection
- **Fitness**: Combination of survival time and food consumption
- **Selection**: Natural selection through survival and reproduction
- **Tracking**: Best agent genome and memory state saved for analysis

---

## ⚙️ Configuration

All parameters are configured through `config.yaml`:

### World Physics
```yaml
world:
  width: 800          # World width in pixels
  height: 600         # World height in pixels
  dt: 0.05           # Time step in seconds
  speed_px_s: 120    # Pixels per second at speed=1.0
```

### Agent Metabolism
```yaml
agents:
  init_count: 300                # Initial population
  max_agents: 300               # Population cap
  initial_life: 50.0            # Starting life energy
  baseline_cost: 1.5            # Life cost per second (idle)
  speed_cost_coeff: 1.0         # Additional cost per unit speed
  steering_cost_coeff: 50.0     # Additional cost per unit steering
  collision_damage: 30.0        # Life lost per collision per second
  reproduction_prob: 0.02       # Reproduction probability per step
  reproduction_min_life: 100.0  # Minimum life to reproduce
  life_transfer: 30.0           # Life given to child
```

### Neural Network
```yaml
brain:
  neurons: 10                           # Hidden layer size
  input_dim: "{{sensor.n_rays * 2}}"    # Auto-calculated (10)
  output_dim: 2                         # Steer and speed
  use_bias: true                        # Include bias terms
  decay_factor: 0.03                    # Weight decay (1-decay_factor)
  memory_inheritance_mutation: 0.02     # Memory inheritance noise
  mutation_constant: 0.01               # Proportional mutation constant
```

### Sensors
```yaml
sensor:
  n_rays: 5          # Number of sensor rays
  fov_deg: 60        # Field of view in degrees
  ray_length: 80     # Maximum sensor range
  mode: distance     # 'distance' or 'binary'
```

### Mutation Rates
```yaml
evolution:
  mutation_std:
    W: 0.005   # Recurrent weights (conservative)
    b: 0.003   # Bias terms (very conservative)
    E: 0.015   # Input weights (exploratory)
    D: 0.015   # Output weights (exploratory)
```

---

## 📊 Data Analysis

### Automatic Data Export
When using `--save-data`, the system automatically generates:

1. **`{prefix}_history.csv`**: Complete fitness evolution data
2. **`{prefix}_best_genome.pkl`**: Best agent's neural network
3. **`{prefix}_config.pkl`**: Configuration used for the run
4. **`{prefix}_network.json`**: Network data for HTML visualizer
5. **Summary plots**: Fitness evolution, population dynamics, network architecture

### Analysis Scripts
- **`summary_plotter.py`**: Comprehensive analysis with matplotlib
- **`export_network_data.py`**: Convert genome data to JSON format
- **`network_visualizer.html`**: Interactive network visualization

### Metrics Tracked
- **Fitness evolution**: Best and average fitness over time
- **Population dynamics**: Population size changes
- **Life statistics**: Average and best life values
- **Food consumption**: Feeding behavior analysis
- **Neural network**: Weight distributions and connectivity

---

## 🏗️ Architecture

### Core Components
- **`main.py`**: Entry point and command-line interface
- **`world.py`**: World physics, agent management, and evolution
- **`agent.py`**: Neural network implementation and genetics
- **`utils.py`**: Utility functions for physics calculations
- **`summary_plotter.py`**: Analysis and visualization tools

### Key Design Principles
- **Vectorized operations**: All computations use NumPy for speed
- **Modular design**: Clean separation between physics, genetics, and visualization
- **Configurable parameters**: Easy experimentation through YAML config
- **Data-driven analysis**: Comprehensive logging and analysis tools

---

## 🔮 Future Directions

- **Advanced sensors**: Vision, communication, or environmental gradients
- **Multi-objective fitness**: Balance between survival, reproduction, and cooperation
- **Speciation**: Mechanisms for evolving distinct behavioral niches
- **Gradient-based learning**: Hybrid evolutionary-gradient optimization
- **Emergent communication**: Agent-to-agent signaling
- **Hierarchical control**: Multi-level neural architectures

---

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run basic simulation**: `python main.py --vis`
3. **Collect data**: `python main.py --save-data --data-prefix test`
4. **Analyze results**: Open `network_visualizer.html` and load the generated JSON
5. **Experiment**: Modify `config.yaml` and observe behavioral changes

The system is designed for easy experimentation and analysis, making it perfect for research into artificial life, neural evolution, and emergent behavior.