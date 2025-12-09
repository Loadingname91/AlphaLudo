# Reinforcement Learning for Ludo: A Curriculum-Based Approach

A systematic framework for training deep reinforcement learning agents to master the game of Ludo through progressive difficulty levels. This project implements a 5-level curriculum that builds from basic movement to full 4-player competitive gameplay, achieving **61% win rate** against random opponents in the final multi-agent challenge.

## Overview

This project explores the application of deep reinforcement learning to Ludo, a complex stochastic multi-agent board game. Rather than jumping directly to the full game complexity, we employ a **curriculum learning approach** that incrementally introduces game mechanics:

- **Level 1**: Single token, no opponent interaction (basic movement)
- **Level 2**: Single token with opponent interactions (captures)
- **Level 3**: Multiple tokens per player (token selection strategy)
- **Level 4**: Full stochastic dice mechanics
- **Level 5**: 4-player multi-agent competition

This structured approach enables the agent to learn fundamental skills before tackling the full game's strategic depth.

## Key Features

- **Curriculum-Based Training**: 5 progressive difficulty levels with clear success metrics
- **Dueling Double DQN**: State-of-the-art deep RL architecture with separate value and advantage streams
- **Potential-Based Reward Shaping (PBRS)**: Theory-grounded reward engineering that preserves optimal policies
- **Comprehensive Evaluation**: Detailed metrics tracking win rates, captures, game lengths, and learning dynamics
- **Modular Architecture**: Clean separation between environments, agents, and training logic
- **Reproducibility**: Seed management and hyperparameter tracking for all experiments

## Results Summary

| Level | Challenge | Target | Achieved | Episodes |
|-------|-----------|--------|----------|----------|
| 1 | Basic Movement | 90% | **95%** | 2,500 |
| 2 | Opponent Interaction | 85% | **90%** | 5,000 |
| 3 | Multi-Token Strategy | 75% | **78%** | 7,500 |
| 4 | Stochastic Dynamics | 62% | **67%** | 10,000 |
| 5 | Multi-Agent Chaos | 52% | **61%** | 15,000 |

The agent demonstrates strong performance across all levels, with the final model achieving **2.4x better than random baseline** (25%) in 4-player games.

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RLagentLudo
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train an agent on a specific level:

```bash
# Level 1: Basic movement
python experiments/level1_train.py --episodes 2500 --eval_freq 500

# Level 5: Full game (4 players, 2 tokens each)
python experiments/level5_train.py --episodes 15000 --eval_freq 1000
```

### Testing

Evaluate a trained model:

```bash
# Test Level 5 agent
python experiments/test_level5.py --checkpoint checkpoints/level5/best_model.pth --num_eval 400
```

### Common Training Arguments

- `--episodes`: Total training episodes
- `--eval_freq`: Evaluation frequency (episodes)
- `--num_eval`: Number of evaluation games
- `--lr`: Learning rate (default: 5e-5)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 128)
- `--buffer_size`: Replay buffer size
- `--device`: Training device (cpu/cuda)

## Project Structure

```
RLagentLudo/
├── experiments/              # Training and testing scripts
│   ├── level1_train.py      # Level 1: Basic movement
│   ├── level2_train.py      # Level 2: With captures
│   ├── level3_train.py      # Level 3: Multi-token
│   ├── level4_train.py      # Level 4: Stochastic
│   ├── level5_train.py      # Level 5: Multi-agent
│   └── test_level*.py       # Evaluation scripts
├── src/rl_agent_ludo/
│   ├── agents/              # Agent implementations
│   │   ├── unifiedDQNAgent.py      # Dueling Double DQN
│   │   ├── baseline_agents.py      # Random agent
│   │   ├── simple_dqn.py           # Basic DQN
│   │   └── trajectoryBuffer.py     # Replay buffer
│   ├── environment/         # Environment wrappers
│   │   ├── level1_simple.py        # Level 1 environment
│   │   ├── level2_interaction.py   # Level 2 environment
│   │   ├── level3_multitoken.py    # Level 3 environment
│   │   ├── level4_stochastic.py    # Level 4 environment
│   │   ├── level5_multiagent.py    # Level 5 environment
│   │   └── unifiedLudoEnv.py       # Base environment
│   ├── ludo/                # Core game logic
│   │   ├── game.py
│   │   ├── player.py
│   │   └── visualizer.py
│   ├── tests/               # Test suite
│   └── utils/               # Utilities
├── checkpoints/             # Trained model checkpoints
│   ├── level1/
│   ├── level2/
│   ├── level3/
│   ├── level4/
│   └── level5/
├── docs/                    # Documentation
├── .projectDescription/     # Research papers and design docs
└── requirements.txt
```

## Agent Architecture

### Dueling Double DQN

The primary agent uses a **Dueling Double DQN** architecture, which combines three powerful techniques:

1. **Dueling Networks**: Separate value and advantage streams for better learning
2. **Double Q-Learning**: Reduces overestimation bias by decoupling action selection and evaluation
3. **Prioritized Experience Replay**: Focuses learning on important transitions

**State Representation (16D):**
- Token positions (4 values: 0-59 for board position, -1 for home/goal)
- Goal completion flags (4 binary values)
- Distance metrics (4 values: remaining distance to goal)
- Positional encoding (4 values: normalized progress)

**Network Architecture:**
- Input: 16D state vector
- Hidden layers: 128x128 (ReLU activation)
- Value stream: Single scalar output
- Advantage stream: N actions (varies by level)
- Aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

### Reward Shaping

The agent uses **Potential-Based Reward Shaping (PBRS)** to guide learning while preserving optimal policies:

- **Win/Loss**: +1000 (win), -1000 (loss)
- **Progress Shaping**: Distance-based potential function
- **Capture Rewards**: +50 (capture), -50 (captured)
- **Goal Completion**: +100 per token

PBRS guarantees that the shaped reward function has the same optimal policy as the original sparse reward, while significantly accelerating learning.

## Curriculum Design

### Level 1: Basic Movement (2,500 episodes)
- **Goal**: Learn to move a single token from start to goal
- **State**: 4D (1 token position, goal flag, distance, progress)
- **Actions**: Move token 0
- **Challenge**: Basic sequential decision-making

### Level 2: Opponent Interaction (5,000 episodes)
- **Goal**: Learn to capture opponents and avoid being captured
- **State**: 8D (player + opponent token states)
- **Actions**: Move token 0
- **Challenge**: Adversarial interaction, risk assessment

### Level 3: Multi-Token Strategy (7,500 episodes)
- **Goal**: Manage 2 tokens simultaneously, strategic token selection
- **State**: 16D (2 tokens × 2 players)
- **Actions**: Move token 0 or token 1
- **Challenge**: Resource allocation, multi-objective optimization

### Level 4: Stochastic Dynamics (10,000 episodes)
- **Goal**: Handle full dice mechanics (1-6 outcomes)
- **State**: 16D
- **Actions**: Move token based on dice roll
- **Challenge**: Partial observability, long-term planning under uncertainty

### Level 5: Multi-Agent Chaos (15,000 episodes)
- **Goal**: Compete against 3 random opponents simultaneously
- **State**: 16D (focused on player's own tokens)
- **Actions**: Token selection with dice
- **Challenge**: Full game complexity, emergent multi-agent dynamics

## Evaluation Metrics

Each level tracks:
- **Win Rate**: Primary success metric
- **Average Reward**: Cumulative episode reward
- **Game Length**: Steps per episode
- **Capture Statistics**: Captures made vs. received
- **Epsilon**: Exploration rate (decays from 1.0 to 0.02)
- **Replay Buffer Size**: Experience collected

Evaluations run every N episodes with 200-400 test games against random opponents.

## Development

### Running Tests

```bash
# Run all tests
pytest src/rl_agent_ludo/tests/

# Run specific test file
pytest src/rl_agent_ludo/tests/test_unified_ludo_env_rewards.py
```

### Code Style

This project follows PEP 8 style guidelines with type hints where applicable.

## Documentation

- `docs/agents/` - Agent methodologies and architectures
- `docs/stateAbstraction/` - State representation techniques
- `docs/gameLogic/` - Game mechanics and physics
- `.projectDescription/Research/` - Academic papers and research references

## Research Foundation

This project builds upon established research in:
- **Curriculum Learning**: Progressive task difficulty for skill acquisition
- **Reward Shaping**: Potential-based reward shaping (Ng et al.)
- **Deep RL**: DQN, Double DQN, Dueling architectures (DeepMind)
- **Multi-Agent RL**: Competitive gameplay and emergent strategies

### Key References

Academic papers and resources can be found in `.projectDescription/Research/`:
- Complexity Analysis and Playing Strategies for Ludo
- Multi-agent Ludo Game Collaborative Path
- Skill Dominance Analysis of Ludo Game
- Strategy Game-Playing with Size-Constrained State Abstraction
- Modern Theory of State Abstraction

## Future Work

Potential extensions and improvements:
- **Self-Play Training**: Train against past versions of the agent
- **Multi-Agent Learning**: Simultaneous training of all players
- **Policy Gradient Methods**: PPO, A3C for continuous improvement
- **Transfer Learning**: Leverage lower-level curricula for faster convergence
- **Opponent Modeling**: Explicit modeling of opponent strategies
- **Human Evaluation**: Testing against human players

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_agent_ludo_curriculum,
  title = {Reinforcement Learning for Ludo: A Curriculum-Based Approach},
  author = {Balegar, Hitesh},
  year = {2025},
  url = {https://github.com/yourusername/RLagentLudo},
  note = {Deep RL with progressive curriculum for multi-agent board games}
}
```

## Acknowledgments

This project builds upon:
- **LudoPy**: Python implementation of Ludo game mechanics
- **DeepMind**: DQN, Double DQN, and Dueling DQN architectures
- **OpenAI**: Reinforcement learning best practices and methodologies
- Existing research on RL applications to board games and curriculum learning
