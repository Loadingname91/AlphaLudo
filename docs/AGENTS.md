# Agent Implementations

This project implements four distinct agents, ranging from random baselines to advanced deep reinforcement learning models. 

Select an agent below to view its detailed documentation, theory, and implementation logic.

## [Random Agent](agents/random/README.md)
**Baseline** | **Win Rate: ~25%**
- Serves as the control group for all experiments.
- Useful for validating environment mechanics.
- **[Theory](agents/random/THEORY.md)** | **[Implementation](agents/random/IMPLEMENTATION.md)**

## [Rule-Based Heuristic Agent](agents/rule_based_heuristic/README.md)
**Expert System** | **Win Rate: ~32-35%**
- Encodes human domain knowledge into priority rules.
- Features a 3-layer decision engine (Instincts $\rightarrow$ Strategy $\rightarrow$ Context).
- **[Theory](agents/rule_based_heuristic/THEORY.md)** | **[Logic Flow](agents/rule_based_heuristic/LOGIC_FLOW.md)** | **[Implementation](agents/rule_based_heuristic/IMPLEMENTATION.md)**

## [Tabular Q-Learning Agent](agents/q_learning/README.md)
**RL (Tabular)** | **Win Rate: ~31-43%**
- Learns to play by abstracting the board into tactical features.
- Uses context-aware reward shaping to adapt its strategy (Winning vs. Losing).
- **[Theory](agents/q_learning/THEORY.md)** | **[State Abstraction](agents/q_learning/STATE_ABSTRACTION.md)** | **[Learning Process](agents/q_learning/LEARNING_PROCESS.md)**

## [Dueling Double DQN Agent](agents/dqn/README.md)
**Deep RL** | **Win Rate: ~32-43%**
- Uses neural networks to approximate Q-values from continuous state vectors.
- Implements advanced stability mechanisms: Double Q-Learning, Dueling Architecture, Prioritized Replay.
- **[Theory](agents/dqn/THEORY.md)** | **[Architecture](agents/dqn/ARCHITECTURE.md)** | **[Learning Process](agents/dqn/LEARNING_PROCESS.md)**

---

## Comparative Results

For a detailed comparison of performance, learning curves, and sample efficiency across all agents, see:
**[Experimental Results & Analysis](EXPERIMENTAL_RESULTS.md)**
