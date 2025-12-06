# Reinforcement Learning Agents for Ludo

An experimental framework for training and evaluating reinforcement learning agents in the game of Ludo. This project implements multiple RL algorithms ranging from tabular methods to deep learning approaches, with a focus on modularity, reproducibility, and empirical analysis.

## Motivation

This project aims to develop a well-documented, comprehensive approach to applying reinforcement learning techniques to the game of Ludo. While several implementations exist for RL in Ludo, many lack detailed documentation, theoretical foundations, and systematic comparisons of different RL approaches. 

My goal is to provide:
- **Clear theoretical foundations** for state abstraction, reward shaping, and agent architectures
- **Comprehensive documentation** of design decisions, implementation details, and experimental methodology
- **Systematic comparison** of multiple RL algorithms (tabular Q-learning, DQN, and advanced variants)
- **Reproducible experiments** with detailed configuration management and result tracking
- **Extensible framework** that allows researchers to easily implement and test new RL approaches

This work builds upon and extends existing research in applying RL to board games, with particular attention to the unique challenges posed by Ludo's stochastic nature, multi-agent dynamics, and strategic complexity.

## Abstract

This project explores the application of reinforcement learning to Ludo, a stochastic multi-agent board game. I implement and compare several RL algorithms including tabular Q-learning, deep Q-networks (DQN), Dueling DQN, and rule-based heuristics. The framework provides a clean interface for experimentation, with support for different reward shaping strategies, state abstractions, and learning algorithms. My unified approach includes both traditional state abstractions and novel egocentric physics-based representations, enabling comprehensive evaluation of RL techniques in this challenging domain.

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

I implement a spectrum of agents ranging from stochastic baselines to deep reinforcement learning models.

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

## References

This project builds upon and extends several existing implementations and research:

### Related Projects

- **LudoPy**: A Python implementation of the Ludo game. This project uses LudoPy as the foundation for the game environment.
  - Repository: [LudoPy on GitHub](https://github.com/SimonKnudsen/LudoPy)

- **Ludo Game AI** (Aurucci et al.): A reinforcement learning project applying Q-learning, SARSA, and DQ-learning to a simplified 2-player, 2-token Ludo variant. This work provided insights into state space design and reward shaping strategies.
  - Repository: [Ludo_Game_AI](https://github.com/raffaele-aurucci/Ludo_Game_AI)
  - Contributors: @raffaele-aurucci, @AngeloPalmieri, @CSSabino

- **AI-Ludo** (Sangrasi): Another RL implementation for Ludo with different state representations and training approaches.
  - Repository: [AI-Ludo](https://github.com/MehranSangrasi/AI-Ludo)

### Academic References

- IEEE Paper on RL for Ludo: Research paper exploring reinforcement learning techniques for Ludo game AI.
  - DOI: [10.1109/10409945](https://ieeexplore.ieee.org/document/10409945/)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_agent_ludo,
  title = {Reinforcement Learning Agents for Ludo: A Well-Documented Approach},
  author = {Balegar, Hitesh},
  year = {2025},
  url = {https://github.com/yourusername/RLagentLudo},
  note = {Built upon LudoPy and inspired by related RL implementations for Ludo}
}
```

### Acknowledgments

I acknowledge the following projects and researchers for their contributions to RL in Ludo:
- The LudoPy project for providing the game environment foundation
- Raffaele Aurucci, Angelo Palmieri, and CSSabino for their work on Ludo Game AI
- Mehran Sangrasi for the AI-Ludo implementation
- All researchers contributing to the academic literature on RL for board games
