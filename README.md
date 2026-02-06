# Quadruped Locomotion: Training and Deployment Pipeline

This repository provides a complete pipeline for training and deploying reinforcement learning locomotion policies on the Unitree Go2 quadruped robot. The project is split into two submodules that handle training and deployment separately.

---

> **⚠️ ACTIVE DEVELOPMENT NOTICE**
>
> This repository is under active development, especially **[quadruped-locomotion-deploy](https://github.com/illusoryTwin/quadruped-locomotion-deploy)**. You may encounter bugs, incomplete features, or breaking changes. They are temporary.

---

## Overview

The pipeline consists of two main components:

1. **[quadruped-locomotion-rl](https://github.com/illusoryTwin/quadruped-locomotion-rl)** - RL training environment using Isaac Lab
2. **[quadruped-locomotion-deploy](https://github.com/illusoryTwin/quadruped-locomotion-deploy)** - Deployment interface for simulation in Mujoco and real robot hardware using DDS communication

### Supported Tasks

Currently implemented locomotion tasks:
- Walking on flat terrain
- Walking on rough terrain
- Climbing upstairs

- Soft compliant policies (compliant locomotion).

## Architecture

```
quadruped-locomotion/
├── quadruped-locomotion-rl/          # Training repository
│   ├── scripts/                      # Training & evaluation
│   │   ├── train.py                  # Main training script
│   │   └── play.py                   # Visualization in Isaac Sim
│   ├── tasks/                        # Environment configurations
│   │   ├── flat_walk_env_cfg.py
│   │   ├── rough_walk_env_cfg.py
│   │   └── stairs_climbing_env_cfg.py
│   ├── modules/                      # Custom RL components
│   │   ├── terrains.py               # Terrain generation
│   │   ├── rewards.py                # Reward functions
│   │   └── curriculums.py            # Training curricula
│   ├── deploy/                       # MuJoCo testing
│   │   ├── configs/                  # Deployment configs
│   │   ├── common/                   # Shared utilities
│   │   └── mujoco/                   # MuJoCo simulator interface
│   └── logs/                         # Training checkpoints & outputs
│
└── quadruped-locomotion-deploy/     # Real robot deployment
    ├── core/                         # Core deployment logic
    │   ├── policy_controller.py      # Policy inference
    │   ├── command_manager.py        # Velocity command handling
    │   └── config.py                 # Configuration dataclasses
    ├── runners/                      # Robot interface
    │   ├── dds_runner.py             # DDS communication loop
    │   └── dds_interface.py          # Low-level DDS interface
    ├── utils/                        # Utilities
    │   └── joint_mapper.py           # Joint order mapping
    └── run.py                        # Main deployment script
```

<!-- ## Complete Pipeline

### Phase 1: Training (Isaac Sim + Isaac Lab)

Train policies in simulation using the RL submodule:

```
Isaac Lab → PPO Training → Trained Policy Checkpoint (.pt)
```

### Phase 2: Testing (MuJoCo)

Validate policies in MuJoCo before real robot deployment:

```
Trained Checkpoint → MuJoCo Simulator → Policy Validation
```

### Phase 3: Deployment (Real Robot)

Deploy validated policies to hardware:

```
Validated Checkpoint + Config → DDS Interface → Unitree Go2 Robot
```

--- -->

## Installation

### Prerequisites

- Ubuntu 20.04+ / 22.04 recommended
- NVIDIA GPU with CUDA support

### Clone Repository

```bash
git clone --recursive git@github.com:illusoryTwin/quadruped-locomotion.git
cd quadruped-locomotion
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### Environment 1: Training Setup (Isaac Sim + Isaac Lab)

Isaac Sim and Isaac Lab are required for RL training.

**Install Isaac Sim and Isaac Lab:**

Follow the official installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

**Dependencies:**
| Package | Version | Installation |
|---------|---------|--------------|
| Isaac Sim | 5.1 | Follow guide above |
| Isaac Lab | 2.3.0 | Installed via Isaac Lab setup |
| RSL-RL | 3.0.1+ | `pip install rsl-rl` |

**Install this package:**
```bash
# Activate Isaac Lab environment
conda activate isaacsim

# Navigate to RL submodule and install
cd quadruped-locomotion-rl
pip install -e .
```

### Environment 2: MuJoCo Testing Setup

For testing policies in MuJoCo simulation before robot deployment:

```bash
# Create conda environment
conda create -n go2_deploy python=3.10 -y
conda activate go2_deploy

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install MuJoCo and dependencies
pip install mujoco pygame numpy pyyaml

# Set up external dependencies
export UNITREE_MUJOCO_PATH=~/unitree_robotics/unitree_mujoco
mkdir -p $(dirname $UNITREE_MUJOCO_PATH)

# Clone Unitree repos
git clone https://github.com/unitreerobotics/unitree_mujoco $UNITREE_MUJOCO_PATH
git clone https://github.com/unitreerobotics/unitree_sdk2_python $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python

# Install SDK
pip install -e $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python

# Make environment variable persistent
echo "export UNITREE_MUJOCO_PATH=$UNITREE_MUJOCO_PATH" >> ~/.bashrc
```

### Environment 3: Real Robot Deployment Setup

For deploying to actual Unitree Go2 hardware:

```bash
# Create deployment environment
conda create -n go2_robot python=3.10 -y
conda activate go2_robot

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Unitree SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python ~/unitree_sdk2_python
pip install -e ~/unitree_sdk2_python

# Install other dependencies
pip install numpy pyyaml
```

---

## Usage Guide

### 1. Training Policies

Train locomotion policies using Isaac Lab:

```bash
# Navigate to RL submodule
cd quadruped-locomotion-rl

# Activate training environment
conda activate isaacsim

# Train flat terrain walking (basic task)
../../IsaacLab/isaaclab.sh -p scripts/train.py \
    --task=go2_walk_flat \
    --num_envs=4096 \
    --max_iterations=5000

# Train rough terrain walking (challenging)
../../IsaacLab/isaaclab.sh -p scripts/train.py \
    --task=go2_walk_rough \
    --num_envs=4096 \
    --max_iterations=5000

# Train stairs climbing (complex task)
../../IsaacLab/isaaclab.sh -p scripts/train.py \
    --task=go2_stairs_climbing \
    --num_envs=4096 \
    --max_iterations=5000
```
<!-- 
**Training outputs:**
- Checkpoints: `logs/rsl_rl/<task_name>/<timestamp>/model_<iter>.pt`
- TensorBoard logs: `logs/rsl_rl/<task_name>/<timestamp>/`
- Configuration: Saved with checkpoint

**Monitor training:**
```bash
tensorboard --logdir=logs/rsl_rl/
``` -->

### 2. Visualize Trained Policy

Preview policy behavior in Isaac Sim before deployment:

```bash
cd quadruped-locomotion-rl
conda activate isaacsim

# Visualize latest checkpoint
../../IsaacLab/isaaclab.sh -p scripts/play.py \
    --task=go2_walk_flat \
    --num_envs=16

# Visualize specific checkpoint
../../IsaacLab/isaaclab.sh -p scripts/play.py \
    --task=go2_walk_flat \
    --num_envs=16 \
    --checkpoint=logs/rsl_rl/unitree_go2_walk/2025-12-29_15-43-52/model_1500.pt
```

### 3. Test in MuJoCo Simulation

Validate policies in MuJoCo simulation before deploying to the real robot.

**Terminal 1 - Launch MuJoCo simulator:**

```bash
# Navigate to the Unitree MuJoCo directory
cd $UNITREE_MUJOCO_PATH

# Activate deployment environment
conda activate go2_deploy

# Launch the simulator with Go2 robot model
python3 simulate.py
```

If you don't have the `UNITREE_MUJOCO_PATH` environment variable set:
```bash
cd ~/unitree_robotics/unitree_mujoco
python3 simulate.py
```

The simulator will open a window showing the Go2 robot in the MuJoCo environment.

**Terminal 2 - Run policy:**

```bash
cd quadruped-locomotion-deploy
conda activate go2_deploy
python run.py
```

The policy will connect to the simulator via DDS and start controlling the robot. You should see the robot executing the trained locomotion policy in the MuJoCo window.

