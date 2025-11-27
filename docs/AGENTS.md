# Agent Implementations: Theory and Results

This document describes the theoretical foundations, implementation details, and experimental results for each agent type implemented in this project.

## Table of Contents

1. [Random Agent](#random-agent)
2. [Rule-Based Heuristic Agent](#rule-based-heuristic-agent)
3. [Tabular Q-Learning Agent](#tabular-q-learning-agent)
4. [Dueling Double DQN Agent](#dueling-double-dqn-agent)

---

## Random Agent

### Theory

The random agent serves as a baseline for comparison. It selects uniformly random actions from the set of valid moves at each turn. In a 4-player game with equal skill opponents, the expected win rate is 25% (1/4 players).

### Implementation

- **Action Selection**: Uniform random sampling from `state.valid_moves`
- **Learning**: None (non-learning baseline)
- **State Representation**: Not used (actions are independent of state)

### Hyperparameters

None (no learning parameters).

### Results Summary

- **Expected Win Rate**: ~25%
- **Use Case**: Baseline validation of environment and training infrastructure

### Key Files

- `src/rl_agent_ludo/agents/random_agent.py`

---

## Rule-Based Heuristic Agent

### Theory

The rule-based heuristic agent uses hand-crafted priority rules to evaluate moves. It implements a two-layer scoring system:

1. **Priority Layer**: High-priority rules (winning, capturing, fleeing) override all other considerations
2. **Strategic Layer**: Additive scoring for progress, star jumps, globe positions, and piece balance

The agent also uses **phase-aware contextual multipliers** that adjust rule priorities based on game phase:
- **Opening**: Prioritize getting pieces out of home
- **Midgame**: Balanced play
- **Closing**: Focus on finishing pieces
- **Critical**: Aggressive play when opponent is close to winning

### Implementation

**Scoring Components**:
- `WIN_MOVE = 1,000,000`: Winning the game
- `CAPTURE_MOVE = 50,000`: Capturing an opponent piece
- `FLEE_MOVE = 50,000`: Escaping from threat
- `HOME_BASE_PROGRESS = 10,000`: Moving in safe zones/home stretch
- `BLOCKADE_CLUSTER = 4,000`: Blocking opponent's cluster
- `FORM_BLOCKADE_MOVE = 3,500`: Forming a blockade
- `GET_OUT_OF_HOME = 7,000`: Exiting home base
- `PROGRESS_SCORE = 1,000`: Basic forward progress
- `STAR_JUMP = 5,500`: Jumping via star
- `GLOBE_HOP = 500`: Landing on safe globe
- `BALANCED_FRONT = 350`: Moving least advanced piece
- `RISK_FACTOR = 800`: Penalty for landing near enemies

**Risk Calculation**: Probabilistic risk based on distance to threatening enemies:
```
risk = Σ (7 - distance) × RISK_FACTOR for all enemies 1-6 steps away
```

**Phase Detection**:
- Critical: Any opponent has 2+ pieces in goal
- Opening: 2+ pieces stuck at home
- Closing: 2+ pieces in home stretch or finished
- Midgame: Default

### Hyperparameters

All scoring weights are hand-tuned constants (see implementation).

### Results Summary

**Performance** (across 3 experiments):
- **Final Win Rate**: 32.0% ± 3.0%
- **Initial Win Rate**: 37.7% ± 1.2% (no learning, consistent performance)
- **Average Episode Length**: 368.3 ± 7.4 steps
- **Best Run**: `rule_based_heuristic_20251120_040448` (35.0% win rate)

**Strengths**: 
- Strong tactical play, good at recognizing immediate opportunities
- Consistent performance (no variance from learning)
- Phase-aware strategy adaptation

**Weaknesses**: 
- No learning or adaptation
- Fixed strategy may be exploitable
- Cannot improve from experience

### Key Files

- `src/rl_agent_ludo/agents/rule_based_heuristic_agent.py`

---

## Tabular Q-Learning Agent

### Theory

Q-Learning is a model-free, off-policy temporal difference learning algorithm. The agent learns an action-value function Q(s,a) representing the expected cumulative reward of taking action a in state s.

**Bellman Equation**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

where:
- α: learning rate
- γ: discount factor
- r: immediate reward
- s': next state

**State Abstraction**: To handle the large state space, we use a context-aware potential-based abstraction:
- **State Tuple**: (P1, P2, P3, P4, Context)
  - P1-P4: Potential categories for each piece (NULL, NEUTRAL, RISK, BOOST, SAFETY, KILL, GOAL)
  - Context: Game context (TRAILING, NEUTRAL, LEADING) based on weighted equity

**Weighted Equity Score**:
```
Score = (Goal×100) + (Corridor×50) + (Safe×10) + Distance
```

Context is determined by comparing our weighted score to the maximum opponent score:
- Trailing: gap < -20
- Leading: gap > 20
- Neutral: otherwise

**Dynamic Reward Scaling**: Rewards are scaled based on context and move potential:
- **Trailing Context**: Boost aggressive moves (kill: 1.5×, boost: 1.2×), reduce conservative (safety: 0.8×, goal: 0.5×)
- **Leading Context**: Boost defensive moves (safety: 2.0×, risk avoidance: 1.5×), reduce aggression (kill: 0.8×)
- **Neutral Context**: No scaling (1.0×)

### Implementation

- **Q-Table**: `defaultdict` mapping state tuples to arrays of 4 Q-values (one per piece)
- **Exploration**: ε-greedy policy with exponential decay
- **State Abstraction**: `LudoStateAbstractor` converts raw board state to tactical tuple
- **Reward Scaling**: Context-aware multipliers applied before Q-update

### Hyperparameters

- `learning_rate`: 0.1 (default)
- `discount_factor`: 0.9 (default)
- `epsilon`: 0.1 (initial exploration rate)
- `epsilon_decay`: 0.9995 (per episode)
- `min_epsilon`: 0.01 (minimum exploration)

### Results Summary

**Performance** (across 8 experiments):
- **Final Win Rate**: 30.9% ± 6.0%
- **Initial Win Rate**: 23.2% ± 2.5%
- **Improvement**: +7.7 percentage points (33% relative improvement)
- **Average Episode Length**: 371.3 ± 4.6 steps
- **Best Run**: `q_learning_context_aware_20251121_230058` (43.0% win rate, 30,000 episodes)

**Learning Curve**:
- Starts near baseline (~23% win rate)
- Shows steady improvement over training
- Convergence observed around 300-400 episodes
- Final performance: ~31% win rate (above random baseline)

**Context Distribution** (final 100 episodes):
- **Trailing**: 56.1% (agent often behind)
- **Neutral**: 26.2% (balanced game state)
- **Leading**: 17.7% (agent ahead)

**Strengths**: 
- Adapts to game context through dynamic reward scaling
- Learns from experience and improves over time
- Interpretable Q-table structure
- Context-aware strategy adaptation

**Weaknesses**: 
- State abstraction may lose important information
- Limited by tabular representation
- Requires careful state design
- High variance in final performance across runs

### Key Files

- `src/rl_agent_ludo/agents/QLearning_agent.py`
- `src/rl_agent_ludo/utils/state_abstractor.py`

---

## Dueling Double DQN Agent

### Theory

**Deep Q-Network (DQN)**: Extends Q-learning to large state spaces using neural networks to approximate Q(s,a). DQN uses experience replay and target networks to stabilize training.

**Double DQN**: Addresses overestimation bias in Q-learning by using separate networks for action selection and evaluation:
```
Target = r + γ Q_target(s', argmax_a' Q_online(s', a'))
```

**Dueling Architecture**: Separates value function V(s) and advantage function A(s,a):
```
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```

This allows the network to learn state values independently of action advantages, improving sample efficiency.

**Prioritized Experience Replay (PER)**: Samples transitions with probability proportional to TD-error magnitude, focusing learning on surprising experiences.

**N-step Returns**: Uses multi-step returns to reduce variance and bias in value estimates.

### Implementation

- **Network Architecture**: Dueling DQN with separate value and advantage streams
- **State Representation**: Orthogonal state abstraction (feature vector for neural network)
- **Replay Buffer**: Prioritized experience replay with importance sampling
- **Target Network**: Updated every 1000 steps (configurable)
- **Loss Function**: Huber loss (smooth L1) with importance sampling weights

### Hyperparameters

- `learning_rate`: 0.0001 (default)
- `gamma`: 0.99 (default)
- `epsilon_start`: 1.0
- `epsilon_end`: 0.01
- `epsilon_decay`: 0.995 (per episode)
- `batch_size`: 32
- `buffer_size`: 80,000
- `target_update_freq`: 1000 steps
- `per_alpha`: 0.6 (prioritization exponent)
- `per_beta_start`: 0.4 (importance sampling exponent, annealed to 1.0)

### Results Summary

**Performance** (across 2 experiments):
- **Final Win Rate**: 32.0% ± 15.6%
- **Initial Win Rate**: 25.5% ± 2.1%
- **Improvement**: +6.5 percentage points (25% relative improvement)
- **Average Episode Length**: 375.0 ± 5.6 steps
- **Best Run**: `dqn_dueling_orthogonal_20251124_235806` (43.0% win rate, 40,000 episodes)

**Learning Curve**:
- Starts near baseline (~25% win rate)
- Shows improvement over training
- Convergence observed around 300 episodes
- Final performance: ~32% win rate (above random baseline)
- High variance suggests need for more training or hyperparameter tuning

**Sample Efficiency**:
- Requires more episodes than tabular Q-learning (40,000 vs 30,000 for best runs)
- Slower initial learning but potential for higher final performance

**Strengths**: 
- Can learn complex strategies
- Handles large state spaces through neural network approximation
- Prioritized replay focuses learning on important experiences
- Dueling architecture improves sample efficiency

**Weaknesses**: 
- Requires more samples than tabular methods
- Hyperparameter sensitive (high variance across runs)
- Less interpretable than tabular Q-learning
- Longer training time required

### Key Files

- `src/rl_agent_ludo/agents/dqn_agent.py`
- `src/rl_agent_ludo/agents/modules/dueling_dqn_network.py`
- `src/rl_agent_ludo/utils/orthogonal_state_abstractor.py`
- `src/rl_agent_ludo/utils/prioritized_replay_buffer.py`

---

## Comparative Analysis

[To be filled with cross-agent comparisons]

### Performance Comparison

| Agent | Final Win Rate | Initial Win Rate | Improvement | Episodes (Best) | Complexity |
|-------|---------------|------------------|-------------|-----------------|------------|
| Random | ~25% | ~25% | N/A | N/A | Low |
| Rule-Based | 32.0% ± 3.0% | 37.7% ± 1.2% | N/A (no learning) | 1,000 | Medium |
| Q-Learning | 30.9% ± 6.0% | 23.2% ± 2.5% | +7.7% | 30,000 | Medium |
| DQN | 32.0% ± 15.6% | 25.5% ± 2.1% | +6.5% | 40,000 | High |

### Learning Characteristics

- **Convergence Speed**: Q-Learning (fastest, ~300 episodes) > DQN (~300 episodes) > Rule-Based (N/A)
- **Final Performance**: DQN (best potential, 43% in best run) ≈ Q-Learning (43% in best run) > Rule-Based (35% best) > Random (25%)
- **Stability**: Rule-Based (most stable, no variance) > Q-Learning (moderate variance) > DQN (high variance)
- **Sample Efficiency**: Rule-Based (immediate) > Q-Learning (30K episodes) > DQN (40K episodes)

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*.

