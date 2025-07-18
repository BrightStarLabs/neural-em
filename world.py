# world.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import utils
from agent import Agent, init_population, step as brain_step, apply_proportional_mutation_and_decay

# Optional Pygame import for draw()
try:
    import pygame

    _PG = True
except ModuleNotFoundError:
    _PG = False


# --------------------------------------------------------------------------- #
# Dataclass holding *all* hyper‑parameters                                    #
# New fields have sane defaults so legacy YAML still loads                    #
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    # ------------------- world geometry & timing --------------------------- #
    width: int
    height: int
    dt: float
    speed_px_s: float

    # ------------------- food                                                #
    food_radius: float
    food_nutrition: float
    food_spawn_prob: float
    food_max: int

    # ------------------- agent ecology                                       #
    agents_init: int
    max_agents: int
    initial_life: float
    baseline_cost: float
    speed_cost_coeff: float
    steering_cost_coeff: float
    collision_damage: float
    reproduction_prob: float
    reproduction_min_life: float
    life_transfer: float

    # ------------------- sensor (fixed, not evolved) ----------------------- #
    n_rays: int = 5
    fov_deg: float = 180.0
    ray_length: float = 150.0
    sensor_mode: str = "distance"  # 'distance' | 'binary'

    # ------------------- brain arch                                          #
    neurons: int = 32
    input_dim: int = 5  # ← will be overwritten by n_rays in __post_init__
    output_dim: int = 2
    use_bias: bool = True
    decay_factor: float = 0.001
    memory_inheritance_mutation: float = 0.1
    mutation_constant: float = 0.01

    # ------------------- mutation σ (for offspring mutations) -------------- #
    mut_W: float = 0.10
    mut_b: float = 0.05
    mut_E: float = 0.20
    mut_D: float = 0.20

    # input_dim is now set by config template resolution


# --------------------------------------------------------------------------- #
#                                World class                                  #
# --------------------------------------------------------------------------- #
class World:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.time = 0.0

        # --------------------- AGENTS ------------------------------------- #
        xy, direc, speed, life = Agent.random_spawn(
            cfg.agents_init, cfg.width, cfg.height, cfg.initial_life
        )
        self.a_xy, self.a_dir, self.a_speed, self.a_life = xy, direc, speed, life
        
        # --------------------- FITNESS TRACKING --------------------------- #
        self.a_age = np.zeros(len(self.a_xy), dtype=np.float32)  # Agent ages
        self.a_food_consumed = np.zeros(len(self.a_xy), dtype=np.float32)  # Food eaten per agent
        self.fitness_history = []  # (time, best_fitness, avg_fitness, population_size)
        self.best_agent_idx = 0  # Index of best performing agent
        self.best_fitness = 0.0  # Best fitness score so far

        # neural genomes + hidden state (one row per agent)
        self.genome, self.hidden = init_population(
            cfg.agents_init,
            cfg.neurons,
            cfg.input_dim,
            cfg.output_dim,
            use_bias=cfg.use_bias,
        )

        # --------------------- FOOD --------------------------------------- #
        self.f_xy = self._spawn_food(cfg.food_max // 2)  # start with half cap

        # Pre‑compute ray angles relative to heading  (shape (k,))
        k = cfg.n_rays
        if k == 1:
            self._ray_rel_angles = np.array([0.0], np.float32)
        else:
            span = np.deg2rad(cfg.fov_deg)
            self._ray_rel_angles = np.linspace(
                -span / 2.0, span / 2.0, k, dtype=np.float32
            )

    # ------------------------------------------------------------------ #
    # Utility to spawn N food pellets randomly                            #
    # ------------------------------------------------------------------ #
    def _spawn_food(self, n: int) -> np.ndarray:
        xy = np.empty((n, 2), np.float32)
        rng = np.random.default_rng()
        xy[:, 0] = rng.uniform(0, self.cfg.width, n)
        xy[:, 1] = rng.uniform(0, self.cfg.height, n)
        return xy

    # ------------------------------------------------------------------ #
    # Sensor: cast rays & build input X  (returns shape (N,k))            #
    # ------------------------------------------------------------------ #
    def _sense(self) -> np.ndarray:
        """
        Enhanced ray‑cast: for each agent and each ray, find closest food and 
        closest agent separately. Returns (N, 2*k) array where:
        - First k values are food distances [0,1]
        - Last k values are agent distances [0,1]
        """
        cfg = self.cfg
        N = len(self.a_xy)
        k = cfg.n_rays
        if N == 0:
            return np.empty((0, k * 2), np.float32)

        # Convenience handles
        W, H = cfg.width, cfg.height
        rng = cfg.ray_length

        # Prepare output with "no hit" defaults
        X = np.zeros((N, k * 2), np.float32)  # [food_rays, agent_rays]
        if cfg.sensor_mode == "distance":
            X.fill(1.0)  # normalised range (1 == no hit)

        # Process food and agents separately
        target_sets = [
            ("food", self.f_xy, 0),      # food targets, output offset 0
            ("agents", self.a_xy, k)     # agent targets, output offset k
        ]

        for target_type, targets, output_offset in target_sets:
            if len(targets) == 0:
                continue

            # Pre‑compute pairwise deltas under torus wrap
            # delta shape (N,T,2)
            delta = targets[None, :, :] - self.a_xy[:, None, :]
            delta[:, :, 0] = (delta[:, :, 0] + W / 2.0) % W - W / 2.0
            delta[:, :, 1] = (delta[:, :, 1] + H / 2.0) % H - H / 2.0

            dist = np.sqrt(np.sum(delta**2, axis=-1))  # (N,T)
            
            # avoid self‑hits for agent targets
            if target_type == "agents":
                dist[dist == 0.0] = np.inf

            # angle to target in world frame
            angle_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])  # (N,T)

            # For each ray index compute inputs
            for j, rel in enumerate(self._ray_rel_angles):
                # world‑frame direction of the ray for every agent  (N,)
                dir_j = (self.a_dir + rel) % (2.0 * math.pi)

                # angle difference to every target  (N,T)
                dtheta = np.abs(np.angle(np.exp(1j * (angle_abs - dir_j[:, None]))))
                # which targets lie inside this narrow ray
                if k == 1:
                    in_beam = dtheta < (np.deg2rad(cfg.fov_deg) / 2.0)
                else:
                    step = np.deg2rad(cfg.fov_deg) / (k - 1)
                    in_beam = dtheta < (step / 2.0)

                # apply range filter
                mask = in_beam & (dist <= rng)

                if not mask.any():
                    continue  # no hits for this ray in whole population

                # nearest hit distance per agent for this ray
                nearest = np.where(mask, dist, np.inf)
                d_min = nearest.min(axis=1)  # (N,)

                if cfg.sensor_mode == "binary":
                    X[:, output_offset + j] = (d_min < np.inf).astype(np.float32)
                else:  # 'distance'
                    norm = d_min / rng
                    norm[norm == np.inf] = 1.0
                    X[:, output_offset + j] = norm.astype(np.float32)

        return X

    # ------------------------------------------------------------------ #
    # Main simulation step                                                #
    # ------------------------------------------------------------------ #
    def step(self):
        cfg = self.cfg
        n_agents = len(self.a_xy)
        if n_agents == 0:
            return

        # ---------- Sense & brain --------------------------------------- #
        X = self._sense()  # (N, 2*k) - food distances + agent distances
        actions = brain_step(self.genome, self.hidden, X)
        steer = np.clip(actions[:, 0], -1.0, 1.0)
        speed_cmd = np.clip(actions[:, 1], 0.0, 1.0)

        # ---------- Update heading & speed ------------------------------ #
        self.a_dir = (self.a_dir + steer * (math.pi / 4.0)) % (2.0 * math.pi)
        self.a_speed = speed_cmd

        # ---------- Kinematics ------------------------------------------ #
        dx = (
            self.a_speed
            * np.cos(self.a_dir)
            * cfg.speed_px_s
            * cfg.dt
        )
        dy = (
            self.a_speed
            * np.sin(self.a_dir)
            * cfg.speed_px_s
            * cfg.dt
        )
        self.a_xy[:, 0] += dx
        self.a_xy[:, 1] += dy
        utils.wrap_xy(self.a_xy, cfg.width, cfg.height)

        # ---------- Metabolism ------------------------------------------ #
        speed_penalty = cfg.speed_cost_coeff * self.a_speed
        steering_penalty = cfg.steering_cost_coeff * np.abs(steer)
        self.a_life -= cfg.dt * (cfg.baseline_cost + speed_penalty + steering_penalty)
        
        # Update agent ages
        self.a_age += cfg.dt

        # ---------- Food collisions ------------------------------------- #
        if len(self.f_xy):
            dist_af = utils.pairwise_dist(self.a_xy, self.f_xy)
            hit = dist_af < (Agent.RADIUS + cfg.food_radius)
            a_idx, f_idx = np.where(hit)
            if len(a_idx):
                life_gain = (
                    np.bincount(a_idx, minlength=n_agents) * cfg.food_nutrition
                )
                self.a_life[: len(life_gain)] += life_gain
                
                # Track food consumption for fitness
                food_consumed = np.bincount(a_idx, minlength=n_agents)
                self.a_food_consumed[: len(food_consumed)] += food_consumed
                
                keep = np.ones(len(self.f_xy), bool)
                keep[f_idx] = False
                self.f_xy = self.f_xy[keep]

        # ---------- Agent collisions ------------------------------------ #
        if n_agents > 1:
            dist_aa = utils.pairwise_dist(self.a_xy, self.a_xy)
            np.fill_diagonal(dist_aa, np.inf)
            collide = dist_aa < (2 * Agent.RADIUS)
            dmg = collide.sum(1).astype(np.float32) * cfg.collision_damage * cfg.dt
            self.a_life -= dmg

        # ---------- Reproduction ---------------------------------------- #
        if len(self.a_xy) < cfg.max_agents:
            fertile = self.a_life >= cfg.reproduction_min_life
            repro_mask = fertile & (
                np.random.rand(len(self.a_xy)) < cfg.reproduction_prob
            )
            for p in np.where(repro_mask)[0]:
                if len(self.a_xy) >= cfg.max_agents:
                    break
                if self.a_life[p] <= cfg.life_transfer:
                    continue

                # parent gives life to child
                self.a_life[p] -= cfg.life_transfer

                # spawn child physical state
                child_xy, child_dir, child_speed, child_life = Agent.spawn_near(
                    self.a_xy[p], cfg.width, cfg.height, cfg.life_transfer
                )

                # duplicate parent genome row and mutate offspring
                self.genome.W = np.vstack(
                    [self.genome.W, self.genome.W[p : p + 1]]
                )
                self.genome.E = np.vstack(
                    [self.genome.E, self.genome.E[p : p + 1]]
                )
                self.genome.D = np.vstack(
                    [self.genome.D, self.genome.D[p : p + 1]]
                )
                if isinstance(self.genome.b, np.ndarray):
                    self.genome.b = np.vstack(
                        [self.genome.b, self.genome.b[p : p + 1]]
                    )

                # apply proportional mutation and decay to offspring (last row) only
                child_idx = len(self.a_xy)  # index of the child we're about to add
                child_genome = type(self.genome)(
                    W=self.genome.W[child_idx:child_idx+1],
                    E=self.genome.E[child_idx:child_idx+1], 
                    D=self.genome.D[child_idx:child_idx+1],
                    b=self.genome.b[child_idx:child_idx+1] if isinstance(self.genome.b, np.ndarray) else self.genome.b
                )
                # combine mutation and decay in one operation
                apply_proportional_mutation_and_decay(child_genome, {
                    "W": cfg.mut_W,
                    "b": cfg.mut_b,
                    "E": cfg.mut_E,
                    "D": cfg.mut_D
                }, cfg.decay_factor, cfg.mutation_constant)
                # copy mutated genome back
                self.genome.W[child_idx:child_idx+1] = child_genome.W
                self.genome.E[child_idx:child_idx+1] = child_genome.E
                self.genome.D[child_idx:child_idx+1] = child_genome.D
                if isinstance(self.genome.b, np.ndarray):
                    self.genome.b[child_idx:child_idx+1] = child_genome.b

                # hidden state for child = inherit from parent with mutation
                parent_hidden = self.hidden[p:p+1].copy()  # get parent's hidden state
                if cfg.memory_inheritance_mutation > 0:
                    # Add Gaussian noise to inherited memory
                    mutation_noise = np.random.normal(
                        0, cfg.memory_inheritance_mutation, parent_hidden.shape
                    ).astype(np.float32)
                    parent_hidden += mutation_noise
                
                self.hidden = np.vstack([self.hidden, parent_hidden])

                # append physical arrays
                self.a_xy = np.vstack([self.a_xy, child_xy])
                self.a_dir = np.concatenate([self.a_dir, child_dir])
                self.a_speed = np.concatenate([self.a_speed, child_speed])
                self.a_life = np.concatenate([self.a_life, child_life])
                
                # append fitness tracking arrays
                self.a_age = np.concatenate([self.a_age, np.zeros(1, dtype=np.float32)])
                self.a_food_consumed = np.concatenate([self.a_food_consumed, np.zeros(1, dtype=np.float32)])

        # ---------- Cull the dead --------------------------------------- #
        alive = self.a_life > 0.0
        self._prune_dead(alive)

        # ---------- Probabilistic food spawn ---------------------------- #
        if (
            len(self.f_xy) < cfg.food_max
            and np.random.rand() < cfg.food_spawn_prob
        ):
            new_food = self._spawn_food(1)
            self.f_xy = (
                np.vstack([self.f_xy, new_food]) if len(self.f_xy) else new_food
            )

        self.time += cfg.dt
        
        # Update fitness tracking
        self._update_fitness()
    
    def save_data(self, filename_prefix="simulation"):
        """Save simulation data to CSV and best genome to pickle"""
        import csv
        import pickle
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save fitness history to CSV
        csv_path = f"data/{filename_prefix}_history.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'time', 'best_fitness', 'avg_fitness', 'population_size',
                'avg_life', 'avg_food', 'best_life', 'best_food'
            ])
            writer.writerows(self.fitness_history)
        
        # Save best genome to pickle
        best_genome = self.get_best_agent_genome()
        if best_genome:
            pkl_path = f"data/{filename_prefix}_best_genome.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(best_genome, f)
        
        # Save config for reference
        cfg_path = f"data/{filename_prefix}_config.pkl"
        with open(cfg_path, 'wb') as f:
            pickle.dump(self.cfg, f)
            
        print(f"Data saved to:")
        print(f"  History: {csv_path}")
        print(f"  Best genome: {pkl_path}")
        print(f"  Config: {cfg_path}")

    # ------------------------------------------------------------------ #
    # Fitness tracking and analysis                                       #
    # ------------------------------------------------------------------ #
    def _update_fitness(self) -> None:
        """Update fitness scores and track best performing agent"""
        if len(self.a_xy) == 0:
            return
            
        # Calculate fitness: combination of age and food consumed
        # Fitness = age + food_consumed * 10 (food is more valuable)
        fitness_scores = self.a_age + self.a_food_consumed * 10.0
        
        # Find best performing agent
        if len(fitness_scores) > 0:
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_agent_idx = best_idx
            
            # Record fitness history every 10 steps
            if len(self.fitness_history) == 0 or self.time - self.fitness_history[-1][0] >= 0.5:
                avg_fitness = np.mean(fitness_scores)
                avg_life = np.mean(self.a_life)
                avg_food = np.mean(self.a_food_consumed)
                best_life = self.a_life[best_idx]
                best_food = self.a_food_consumed[best_idx]
                self.fitness_history.append((
                    self.time,
                    best_fitness,
                    avg_fitness,
                    len(self.a_xy),
                    avg_life,
                    avg_food,
                    best_life,
                    best_food
                ))
                
                # Limit history size to prevent memory issues
                if len(self.fitness_history) > 1000:
                    self.fitness_history = self.fitness_history[-800:]
    
    def get_best_agent_genome(self):
        """Get the genome of the best performing agent"""
        if len(self.a_xy) == 0:
            return None
        
        # Ensure best_agent_idx is valid
        if self.best_agent_idx >= len(self.a_xy):
            self.best_agent_idx = 0
            
        return {
            'W': self.genome.W[self.best_agent_idx],
            'E': self.genome.E[self.best_agent_idx],
            'D': self.genome.D[self.best_agent_idx],
            'b': self.genome.b[self.best_agent_idx] if isinstance(self.genome.b, np.ndarray) else self.genome.b,
            'hidden_state': self.hidden[self.best_agent_idx],
            'fitness': self.a_age[self.best_agent_idx] + self.a_food_consumed[self.best_agent_idx] * 10.0,
            'age': self.a_age[self.best_agent_idx],
            'food_consumed': self.a_food_consumed[self.best_agent_idx]
        }

    # ------------------------------------------------------------------ #
    # Helper: drop rows where agent died                                 #
    # ------------------------------------------------------------------ #
    def _prune_dead(self, alive_mask: np.ndarray) -> None:
        self.a_xy = self.a_xy[alive_mask]
        self.a_dir = self.a_dir[alive_mask]
        self.a_speed = self.a_speed[alive_mask]
        self.a_life = self.a_life[alive_mask]
        self.hidden = self.hidden[alive_mask]
        self.genome.W = self.genome.W[alive_mask]
        self.genome.E = self.genome.E[alive_mask]
        self.genome.D = self.genome.D[alive_mask]
        if isinstance(self.genome.b, np.ndarray):
            self.genome.b = self.genome.b[alive_mask]
        
        # Update fitness tracking arrays
        self.a_age = self.a_age[alive_mask]
        self.a_food_consumed = self.a_food_consumed[alive_mask]

    # ------------------------------------------------------------------ #
    # Draw (only if Pygame available)                                     #
    # ------------------------------------------------------------------ #
    def draw(self, screen):
        if not _PG:
            return
        screen.fill((0, 0, 0))
        
        # Draw sensor rays first (so they appear behind agents)
        self._draw_rays(screen)
        
        # food
        for pos in self.f_xy:
            pygame.draw.circle(
                screen, (0, 180, 0), pos.astype(int), int(self.cfg.food_radius)
            )
        # agents
        for pos, ang in zip(self.a_xy, self.a_dir):
            pygame.draw.circle(screen, (50, 120, 255), pos.astype(int), Agent.RADIUS)
            end = pos + np.array([math.cos(ang), math.sin(ang)]) * Agent.RADIUS
            pygame.draw.line(screen, (255, 255, 255), pos.astype(int), end.astype(int), 2)

    def _draw_rays(self, screen):
        """Draw sensor rays for all agents with different colors for food vs agent detections"""
        if not _PG or len(self.a_xy) == 0:
            return
            
        cfg = self.cfg
        
        # Get current sensor readings
        sensor_data = self._sense()  # Shape: (N, 2*k) where first k=food, last k=agents
        k = cfg.n_rays
        
        # Draw rays for each agent
        for i, (pos, heading) in enumerate(zip(self.a_xy, self.a_dir)):
            # Draw each ray
            for j, rel_angle in enumerate(self._ray_rel_angles):
                # Ray direction in world coordinates
                ray_angle = heading + rel_angle
                ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])
                
                # Ray endpoint
                ray_end = pos + ray_dir * cfg.ray_length
                
                # Ensure ray stays within screen bounds for drawing
                ray_end[0] = np.clip(ray_end[0], 0, cfg.width)
                ray_end[1] = np.clip(ray_end[1], 0, cfg.height)
                
                # Get sensor values for this ray
                food_distance = sensor_data[i, j] if i < len(sensor_data) else 1.0
                agent_distance = sensor_data[i, j + k] if i < len(sensor_data) else 1.0
                
                # Determine ray color based on what's detected
                if food_distance < 0.9:  # Food detected (close)
                    color = (0, 255, 0)  # Green for food
                    thickness = 2
                elif agent_distance < 0.9:  # Agent detected (close)
                    color = (255, 100, 100)  # Red for agents
                    thickness = 2
                else:  # Nothing detected
                    color = (60, 60, 60)  # Dark gray for empty
                    thickness = 1
                
                # Draw ray as a line
                pygame.draw.line(
                    screen, 
                    color,
                    pos.astype(int), 
                    ray_end.astype(int), 
                    thickness
                )