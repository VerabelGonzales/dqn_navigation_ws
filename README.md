# DQN Robot Navigation — TurtleBot3

Autonomous navigation system for TurtleBot3 Burger using Deep Q-Network (DQN). The agent learns to reach randomly generated goals in a 4×4 m Gazebo simulation world while avoiding obstacles, using only LiDAR and odometry data.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [State Design](#state-design)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Hyperparameters](#hyperparameters)
- [Training Curves & Results](#training-curves--results)
- [Installation & Usage](#installation--usage)

---

## Architecture Overview

```
Gazebo Simulation (TurtleBot3 Burger)
        │  /scan (LaserScan)
        │  /odom (Odometry)
        ▼
  TurtleBot3Env         ← ROS2 node, wraps sim I/O
        │
  StateProcessor        ← bins LiDAR into 10 sectors + nav features
        │
  DQNAgent              ← Double DQN with sklearn MLPRegressor
  (256 → 256 → 128)     ← Experience replay + target network
        │
  train_node / test_node ← episode loop, reset logic, logging
```

The Q-network is implemented using `sklearn.neural_network.MLPRegressor` with `warm_start=True` and `partial_fit`, enabling incremental online learning without full retraining each step. A separate **target network** is updated via deep copy every `target_update_freq` replay steps to stabilize training.

---

## State Design

**State vector size: 12**

| Index | Feature | Description |
|-------|---------|-------------|
| 0–9 | `lidar_bins[0..9]` | 360° LiDAR scan compressed into 10 equal angular bins. Each bin holds the **minimum** distance reading within that sector. Clipped to [0, 3.5] m. |
| 10 | `goal_distance` | Euclidean distance to goal (m), computed in `odom_callback` from `/odom`. |
| 11 | `goal_angle` | Bearing to goal relative to robot heading (rad), normalized to [−π, π]. |

**LiDAR preprocessing:** raw scan rays (360 total) → 10 bins × 36 rays each → min pooling per bin. Max range capped at 3.5 m; `Inf` → 3.5 m; `NaN` → 0.0 m.

**Goal source of truth:** `goal_distance` and `goal_angle` are calculated exclusively inside `odom_callback` using `self.goal_position`, the same variable that controls the visual marker in Gazebo. This prevents any divergence between what the code detects and what appears on screen.

---

## Action Space

**7 discrete actions** — no backward movement.

| ID | Linear vel (m/s) | Angular vel (rad/s) | Description |
|----|-----------------|---------------------|-------------|
| 0 | 0.26 | 0.00 | Forward (fast) |
| 1 | 0.00 | +0.75 | Rotate left |
| 2 | 0.00 | −0.75 | Rotate right |
| 3 | 0.18 | +0.50 | Forward + left |
| 4 | 0.18 | −0.50 | Forward + right |
| 5 | 0.00 | +1.50 | Rotate left (fast) |
| 6 | 0.00 | −1.50 | Rotate right (fast) |

Actions 1, 2, 5, 6 are classified as **pure rotations** and tracked for oscillation and rotation penalties in the reward function.

---

## Reward Function

The reward is designed to maximize exploratory behavior early in training while still penalizing dangerous proximity to obstacles. There is **no per-step time penalty** — penalizing time causes the agent to learn to rotate in place rather than navigate.

### Per-step reward (non-terminal)

| Component | Formula | Range | Weight | Rationale |
|-----------|---------|-------|--------|-----------|
| Distance progress | `(d_prev − d_curr) × 15` | unbounded | ×1 | Dense main signal — rewards every meter gained toward goal |
| Yaw alignment | `(1 − 2·|θ_goal|/π) × 0.5` | [−0.5, +0.5] | ×0.5 | Gentle orientation guide, does not dominate |
| Obstacle proximity | weighted decay if any ray < 0.30 m | [−0.9, 0] | ×0.3 | Only activates near real danger |
| Forward bonus | +0.2 if linear > 0, else −0.1 | {−0.1, +0.2} | — | Small incentive to use translating actions |
| Oscillation penalty | −1.0 if action sequence A→B→A where (A,B) is a reversal pair | {−1.0, 0} | — | Discourages left-right-left jitter |
| Rotation penalty | −0.5 if last 5 actions are all pure rotations | {−0.5, 0} | — | Discourages spinning in place |

### Terminal rewards

| Event | Reward |
|-------|--------|
| **Goal reached** (`goal_distance < 0.20 m`) | `+200.0` added to step reward |
| **Collision** (>3 LiDAR rays < 0.25 m) | `−50.0` (episode ends) |

### Obstacle reward detail

The proximity penalty uses a **directional weighting** scheme that gives higher weight to obstacles directly in front of the robot (cosine⁶ angular profile). Only rays in the ±90° frontal sector are considered, and only those within 0.30 m. The base function returns [−3, 0]; the external weight of ×0.3 brings the effective range to [−0.9, 0].

### Episode termination

| Condition | Reset type |
|-----------|-----------|
| Collision detected | `reset_full()` — robot respawned at origin + new goal |
| Goal reached | `reset_goal_only()` — robot stays, new goal generated |
| Timeout (1500 steps) | `reset_goal_only()` — robot stays, new goal generated |

---

## Hyperparameters

### DQN Agent

| Parameter | Value | Description |
|-----------|-------|-------------|
| `state_size` | 12 | Input dimension |
| `action_size` | 7 | Output dimension (one Q-value per action) |
| `learning_rate` | 0.0003 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `epsilon_decay_steps` | 6000 | Steps to decay ε via exponential schedule |
| `memory_size` | 20 000 | Replay buffer capacity (FIFO deque) |
| `batch_size` | 64 | Samples per replay update |
| `target_update_freq` | 200 | Replay steps between target network syncs |

**Epsilon decay schedule** (exponential, same as ROBOTIS reference):

```
ε(t) = ε_min + (1 − ε_min) × exp(−t / decay_steps)
```

At t = 4 000 replay steps, ε ≈ 0.10. At t = 6 000, ε ≈ 0.05 (minimum).

### Network Architecture

```
Input (12)  →  Dense 256 (ReLU)  →  Dense 256 (ReLU)  →  Dense 128 (ReLU)  →  Output (7)
```

Regularization: L2 weight decay `α = 0.0001`. Optimizer: Adam.

### Training Loop

| Parameter | Value |
|-----------|-------|
| Total episodes | 220 |
| Max steps / episode | 1 500 |
| Replay starts after | 64 samples in buffer |
| Replay called | every step once buffer ≥ batch_size |
| Model saved at | ep 25, 50, 75, 100, 150, 200, final |

---

## Training Curves & Results

<img src="https://github.com/VerabelGonzales/dqn_navigation_ws/blob/main/models/results_20260219_180800/training_results.png" alt="Acoples">
<p style="margin-top:10px; font-size: 16px;"><strong>Figura 1.</strong> Acoples</p>
<br>

### Reading the curves

**Episode Rewards (top-left):** High variance throughout — expected with epsilon-greedy exploration. Positive rewards dominate from ~ep 50 onward, indicating the agent learns to reach goals rather than just avoid collisions.

**Moving Average Reward — window 20 (top-right):** Clear upward trend from 0 to ~200 by ep 75, confirming genuine learning. A mid-training dip (ep 125–150) is typical as epsilon drops and the agent transitions from exploration to exploitation — Q-values are recalibrating.

**Steps per Episode (mid-left):** Early episodes often hit the 600-step old limit (shown as dashed line). After ep 100, most episodes complete well under the limit, meaning the agent reaches the goal or collides quickly rather than wandering.

**Episode Outcomes — cumulative % (mid-right):**

| Phase | Observation |
|-------|-------------|
| ep 0–30 | Timeout-dominated (~80%) — robot explores but rarely finds goal |
| ep 30–80 | Collision rate rises as ε drops — agent starts committing to actions |
| ep 80–220 | Success rate climbs steadily to ~45%, collision stabilizes ~45%, timeout drops to ~10% |

**Final Distance to Goal (bottom-left):** Majority of episodes end at distance < 0.5 m from ep 100 onward. Spikes indicate difficult random goal placements.

**Min Distance Achieved (bottom-right):** The agent consistently gets within 0.5 m of the goal from ep 80+, and frequently touches the 0.20 m threshold (green dashed line), confirming goal-reaching capability.

### Summary metrics at end of training (ep 220)

| Metric | Value |
|--------|-------|
| Success rate | ~45% |
| Collision rate | ~45% |
| Timeout rate | ~10% |
| Moving avg reward (last 20 ep) | ~200 |

---

## Installation & Usage

### Requirements

- ROS 2 Humble
- Gazebo (Harmonic/Ionic)
- TurtleBot3 packages (`turtlebot3_gazebo`, `turtlebot3_msgs`)
- Python: `numpy`, `scikit-learn`, `matplotlib`

### Training

```bash
ros2 run dqn_robot_nav train_node
```

Results are saved to `results_YYYYMMDD_HHMMSS/`:
- `training_log.txt` — per-episode CSV log
- `final_statistics.txt` — summary metrics
- `training_results.png` — training curves
- `model_ep{N}.pkl` — periodic checkpoints
- `model_final.pkl` — final model

### Evaluation

```bash
ros2 run dqn_robot_nav test_node --ros-args \
  -p model_path:=/path/to/results_YYYYMMDD_HHMMSS/model_final.pkl \
  -p n_episodes:=10 \
  -p stage:=1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | *(required)* | Path to `.pkl` model file |
| `n_episodes` | 10 | Number of evaluation episodes |
| `max_steps` | 1500 | Step limit per episode |
| `stage` | 1 | Gazebo world stage number |

The test node runs in fully greedy mode (`ε = 0`) and prints a result summary with success rate, average reward, and collision/timeout breakdown.

[def]: /models/results_20260219_180800/training_results.png