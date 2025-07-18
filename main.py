# main.py
"""
Entry‑point for Emergent‑Agents simulation.
Loads YAML config → Cfg dataclass, seeds NumPy RNG, then runs either
headless or Pygame‑visual mode.

Usage
-----
Headless, default cfg:
    python main.py
Visual mode:
    python main.py --vis
Re‑run with deterministic RNG:
    python main.py --seed 42
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import yaml
import numpy as np

from world import World, Cfg

try:
    import pygame

    _PG = True
except ModuleNotFoundError:
    _PG = False


# --------------------------------------------------------------------------- #
# YAML → Cfg loader                                                           #
# --------------------------------------------------------------------------- #
def load_cfg(path: str = "config.yaml") -> Cfg:
    """
    Read YAML, flatten all first‑level sections, rename keys so they align
    with the `Cfg` dataclass, then construct and return that dataclass.

    The loader is tolerant: if a YAML block is missing, defaults defined in
    `Cfg` fill the gaps.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Flatten sections that might exist
    flat: dict = {}
    for section in ("world", "food", "agents", "sensor", "brain"):
        flat.update(raw.get(section, {}))

    # Evolution → rename mutation_std.*  → mut_W / mut_b / mut_E / mut_D
    evo = raw.get("evolution", {})
    for k, v in evo.get("mutation_std", {}).items():
        flat[f"mut_{k}"] = v

    # Resolve template references like "{{sensor.n_rays}}" or "{{sensor.n_rays * 2}}"
    def resolve_template(value, context):
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            ref = value[2:-2].strip()
            if ref == "sensor.n_rays":
                return context.get("n_rays", 5)
            elif ref == "sensor.n_rays * 2":
                return context.get("n_rays", 5) * 2
        return value

    # Apply template resolution
    for key, value in flat.items():
        flat[key] = resolve_template(value, flat)

    # Mapping from YAML keys → dataclass field names
    rename = {
        # food
        "radius": "food_radius",
        "nutrition": "food_nutrition",
        "spawn_prob": "food_spawn_prob",
        "max_food": "food_max",
        # agents
        "init_count": "agents_init",
        # sensor
        "mode": "sensor_mode",
    }
    for old, new in rename.items():
        if old in flat:
            flat[new] = flat.pop(old)

    # dataclass will raise if any required key still missing
    return Cfg(**flat)


# --------------------------------------------------------------------------- #
# Runners                                                                     #
# --------------------------------------------------------------------------- #
def run_headless(cfg: Cfg, steps: int = 2000):
    world = World(cfg)
    for i in range(steps):
        world.step()
        if i % 50 == 0:
            print(
                f"t={world.time:6.1f}s | agents={len(world.a_xy):3d} | "
                f"food={len(world.f_xy):3d}"
            )
        time.sleep(0.005)
    return world


def run_vis(cfg: Cfg):
    if not _PG:
        print("Pygame not installed")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Emergent Agents – Neural Evolution")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    world = World(cfg)
    running = True
    frame_count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step()
        world.draw(screen)

        agent_count = len(world.a_xy)
        food_count = len(world.f_xy)
        if frame_count % 60 == 0:
            print(f"t={world.time:6.1f}s | agents={agent_count:3d} | food={food_count:3d}")

        pygame.display.flip()
        clock.tick(60)
        frame_count += 1

    pygame.quit()
    return world





# --------------------------------------------------------------------------- #
# CLI entry                                                                    #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", help="Run with Pygame GUI")
    parser.add_argument("--save-data", action="store_true", help="Save simulation data to files")
    parser.add_argument("--data-prefix", default="simulation", help="Prefix for data files")
    parser.add_argument("--cfg", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, help="Deterministic NumPy seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if not Path(args.cfg).is_file():
        print(f"Config file '{args.cfg}' not found.")
        sys.exit(1)

    cfg = load_cfg(args.cfg)
    
    # Run simulation
    if args.vis:
        world = run_vis(cfg)
    else:
        world = run_headless(cfg)
    
    # Save data if requested
    if args.save_data:
        world.save_data(args.data_prefix)
        
        # Auto-run summary plotter with JSON export
        try:
            import subprocess
            import sys
            print("\n🔄 Running summary plotter with JSON export...")
            result = subprocess.run([
                sys.executable, "summary_plotter.py", 
                "--data-prefix", args.data_prefix,
                "--export-json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Summary plots and JSON export completed successfully!")
            else:
                print(f"⚠️  Summary plotter finished with warnings: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not run summary plotter: {e}")
            print("   You can run it manually: python summary_plotter.py --export-json")


if __name__ == "__main__":
    main()