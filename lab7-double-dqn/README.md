# Lab 7: Double DQN (Reinforcement Learning)

Trains a Double/Dueling DQN agent to solve the CartPole-v1 control task using Gymnasium, with experience replay and a target network for stable Q-learning.

## Contents (`lab7_double_dqn_cartpole.ipynb`)

### Model — `DQN`
A configurable Q-network supporting **Dueling DQN** architecture (`enable_dueling_dqn=True`): the shared feature layer branches into a value stream `V(s)` and an advantage stream `A(s,a)`, which are combined to produce Q-values — this decouples state-value estimation from action-advantage estimation for more stable learning.

### Experience Replay — `ReplayMemory`
A `deque`-backed replay buffer (configurable max length) that stores transitions and supports random mini-batch sampling, decorrelating consecutive training samples.

### Agent — `Agent`
Wraps the environment, network, and training loop, with hyperparameters:
- **Environment:** `CartPole-v1`
- **Learning rate:** 0.001
- **Discount factor (γ):** 0.99
- **Target network sync rate:** every 100 steps
- **Replay memory size:** 100,000
- **Mini-batch size:** 64
- **Epsilon-greedy exploration:** starts at 1.0, decays by 0.9995 per step

### Training & Evaluation
- `train_cartpole1()` runs the full training loop, saving the trained model and a training-progress plot (`runs/cartpole1.png`, `runs/cartpole1.pt`) to the `runs/` directory.
- `test_cartpole1()` loads the trained model and evaluates it in the environment (rendered via `rgb_array` mode for visualization/animation).

## Tech Stack
- Python, PyTorch
- Gymnasium (`CartPole-v1` environment)
- Matplotlib (training curves, animation)

## Requirements
```bash
pip install gymnasium torch matplotlib
```

## Usage
Run all cells in `lab7_double_dqn_cartpole.ipynb` sequentially:
1. Trains the agent (`train_cartpole1()`), saving checkpoints and plots to `runs/`
2. Evaluates the trained agent (`test_cartpole1()`) on the environment

## Notes
- Both standard Double DQN and Dueling DQN mechanisms are implemented; the dueling architecture is toggled via the `enable_dueling_dqn` flag on the `DQN` class.
- Several sections (network sync loop, action selection) were originally fill-in-the-blank exercises, completed as part of this assignment.