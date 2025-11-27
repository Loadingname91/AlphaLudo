# Reinforcement Learning Agents for Ludo

An experimental framework for training and evaluating reinforcement learning agents in the game of Ludo. This project implements multiple RL algorithms ranging from tabular methods to deep learning approaches, with a focus on modularity, reproducibility, and empirical analysis.

## Abstract

This project explores the application of reinforcement learning to Ludo, a stochastic multi-agent board game. We implement and compare several RL algorithms including tabular Q-learning, deep Q-networks (DQN), and rule-based heuristics. The framework provides a clean interface for experimentation, with support for different reward shaping strategies, state abstractions, and learning algorithms.

## Features

- **Multiple Agent Types**: Random baseline, rule-based heuristic, tabular Q-learning, and Dueling Double DQN
- **Modular Architecture**: Clean separation between environment, agents, reward shaping, and training logic
- **State Abstraction**: Context-aware potential-based state representation for tabular methods
- **Reward Shaping**: Support for sparse, dense, and context-aware reward schemas
- **Comprehensive Metrics**: Episode-level and step-level metrics with JSON/CSV export
- **Reproducibility**: Seed management and configuration-based experiments

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RLagentLudo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

Train a Q-Learning agent with default configuration:

```bash
python -m src.rl_agent_ludo.main --config configs/default_config.yaml
```

### Custom Experiment

Run a custom experiment with specific parameters:

```bash
python -m src.rl_agent_ludo.main \
    --config configs/default_config.yaml \
    --experiment-name my_experiment \
    --num-episodes 2000
```

## Project Structure

```
RLagentLudo/
├── src/rl_agent_ludo/
│   ├── agents/              # Agent implementations
│   │   ├── random_agent.py
│   │   ├── rule_based_heuristic_agent.py
│   │   ├── QLearning_agent.py
│   │   └── dqn_agent.py
│   ├── environment/         # Environment wrapper and reward shaping
│   ├── metrics/             # Metrics collection and logging
│   ├── trainer/             # Training loop orchestrator
│   └── utils/               # State representation, config loading, etc.
├── configs/                 # YAML configuration files
├── docs/                    # Documentation
├── results/                 # Experimental results (organized by agent type)
├── tests/                   # Test suite
└── requirements.txt
```

## Implemented Agents

We implement a spectrum of agents ranging from stochastic baselines to deep reinforcement learning models.

| Agent | Type | Description | Documentation |
|-------|------|-------------|---------------|
| **Random** | Baseline | Selects random valid moves. Used for environment validation. | [Docs](docs/agents/random/README.md) |
| **Rule-Based** | Heuristic | Uses human-crafted priority rules (Instincts -> Strategy -> Context). | [Docs](docs/agents/rule_based_heuristic/README.md) |
| **Q-Learning** | Tabular RL | Learns policies over an abstract state space (State Abstraction). | [Docs](docs/agents/q_learning/README.md) |
| **Dueling DQN** | Deep RL | Dueling Double DQN with Prioritized Experience Replay (PER). | [Docs](docs/agents/dqn/README.md) |

> 📚 **Detailed Analysis**: See [AGENTS.md](docs/AGENTS.md) for theory, architecture, and detailed implementation notes for each agent.

## Configuration

Experiments are configured via YAML files. Key configuration sections:

- `experiment`: Experiment name and output directory
- `agent`: Agent type and hyperparameters
- `training`: Number of episodes, logging intervals, etc.
- `environment`: Reward schema, player ID, seed

See `configs/default_config.yaml` for a complete example.

## Results

Experimental results are organized by agent type in the `results/` directory:

- `results/dqn/` - Deep Q-Network experiments
- `results/q_learning/` - Tabular Q-Learning experiments
- `results/rule_based_heuristic/` - Heuristic agent experiments
- `results/random/` - Baseline experiments

Each experiment directory contains:
- Episode-level metrics (JSON and CSV)
- Score debug logs (if enabled)
- Partial snapshots at checkpoints

See `docs/EXPERIMENTAL_RESULTS.md` for detailed analysis and `docs/AGENTS.md` for agent-specific theory and results.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines.

## Documentation

- `docs/AGENTS.md` - Detailed theory and results for each agent type
- `docs/EXPERIMENTAL_RESULTS.md` - Comparative analysis and methodology
- `docs/EXTENDING_AGENTS.md` - Guide for creating and registering custom agents
- `docs/VISUALIZATION_README.md` - Board visualization tools
- `docs/THEORY.md` - Theoretical foundations
- `results/README.md` - Results directory structure

## Extending the Framework

To add your own agent types, see `docs/EXTENDING_AGENTS.md` for a complete guide. The framework supports custom agents through the `AgentRegistry.register_agent()` method, allowing you to add new agents without modifying core code.

Quick example:
```python
from rl_agent_ludo.agents.base_agent import Agent
from rl_agent_ludo.agents.agent_registry import AgentRegistry

class MyAgent(Agent):
    # Implement required methods
    pass

AgentRegistry.register_agent('my_agent', MyAgent)
```

See `examples/custom_agent_example.py` for a working example.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_agent_ludo,
  title = {Reinforcement Learning Agents for Ludo},
  author = {Balegar, Hitesh},
  year = {2025},
  url = {https://github.com/yourusername/RLagentLudo}
}
```
