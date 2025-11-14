# RL Agent Ludo

A research-grade experimental framework for training and comparing reinforcement learning agents in the game of Ludo.

## Overview

This project implements a modular, hierarchical approach to reinforcement learning for Ludo, progressing from a simple random baseline to advanced search-based agents. The architecture is designed for modularity, reproducibility, and rigorous empirical analysis.

## Architecture

The system is built around six key pillars:

1. **LudoEnv** - Environment abstraction layer (HAL) wrapping the ludopy library
2. **State** - Immutable data transfer object for state representation
3. **RewardShaper** - Strategy pattern for reward shaping
4. **Agent** - Abstract interface and implementations (Random, Q-Learning, DQN, PPO, MCTS)
5. **MetricsTracker** - Lightweight metrics collection
6. **Trainer** - Main training orchestrator

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RLagentLudo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train a RandomAgent (Phase 0 baseline):

```bash
python -m src.rl_agent_ludo.main --config configs/default_config.yaml
```

### Configuration

Configuration is done via YAML files. See `configs/default_config.yaml` for an example.

Key configuration options:
- `experiment.name`: Experiment identifier
- `agent.type`: Agent type (random, tabular_q, dqn, ppo, mcts)
- `training.num_episodes`: Number of training episodes
- `environment.reward_schema`: Reward shaping strategy (sparse, dense, decoupled-ila)

### Custom Experiment

```bash
python -m src.rl_agent_ludo.main \
    --config configs/default_config.yaml \
    --experiment-name my_experiment \
    --num-episodes 2000
```

## Project Structure

```
RLagentLudo/
├── src/
│   └── rl_agent_ludo/
│       ├── agents/          # Agent implementations
│       ├── environment/     # LudoEnv and reward shaping
│       ├── metrics/         # MetricsTracker
│       ├── trainer/         # Trainer orchestrator
│       └── utils/           # State DTO, config loader
├── configs/                 # Configuration files
├── tests/                   # Test suite
├── results/                 # Training outputs (metrics, models)
└── requirements.txt
```

## Implementation Phases

- **Phase 0**: RandomAgent baseline (~25% win rate) ✓
- **Phase 1**: Tabular Q-Learning (manual state abstraction)
- **Phase 1.5**: TD(λ) with eligibility traces
- **Phase 2**: Deep Q-Network (neural networks)
- **Phase 3**: PPO (on-policy policy gradient)
- **Phase 4**: MCTS (AlphaLudo-style)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines.

## License

[Your License Here]

## References

See `.projectDescription/implementationPlan.md` for detailed implementation plan and research references.
