# Theoretical Foundations

This document outlines the theoretical foundations underlying the reinforcement learning approaches implemented in this project.

## Reinforcement Learning Basics

### Markov Decision Process (MDP)

Ludo can be modeled as a Markov Decision Process (MDP) with:
- **State Space S**: All possible board configurations
- **Action Space A**: Valid moves for each piece (0-3)
- **Transition Probability P(s'|s,a)**: Stochastic due to dice rolls and opponent actions
- **Reward Function R(s,a,s')**: Shaped rewards based on game events
- **Discount Factor γ**: Future reward discounting (typically 0.9-0.99)

### Value Functions

**State Value Function V(s)**: Expected cumulative reward from state s:
```
V(s) = E[Σ γᵗ rₜ | s₀ = s]
```

**Action Value Function Q(s,a)**: Expected cumulative reward from state s taking action a:
```
Q(s,a) = E[Σ γᵗ rₜ | s₀ = s, a₀ = a]
```

### Bellman Equations

**Bellman Equation for V(s)**:
```
V(s) = Σ P(s'|s,a) [R(s,a,s') + γV(s')]
```

**Bellman Equation for Q(s,a)**:
```
Q(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ max Q(s',a')]
```

## Q-Learning

### Algorithm

Q-Learning is a model-free, off-policy temporal difference learning algorithm:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

where:
- α: learning rate
- r: immediate reward
- γ: discount factor
- s': next state

### Convergence

Under certain conditions (sufficient exploration, finite state/action spaces), Q-learning converges to the optimal Q-function Q*.

### State Abstraction

For large state spaces, we use state abstraction to reduce dimensionality:
- **Potential-Based Abstraction**: Classify each piece's move outcome into categories (NULL, NEUTRAL, RISK, BOOST, SAFETY, KILL, GOAL)
- **Context**: Global game state (TRAILING, NEUTRAL, LEADING) based on weighted equity

## Deep Q-Networks (DQN)

### Function Approximation

DQN uses neural networks to approximate Q(s,a) for large/continuous state spaces:
```
Q(s,a; θ) ≈ Q*(s,a)
```

where θ are network parameters.

### Key Techniques

**Experience Replay**: Store transitions (s,a,r,s') in a buffer and sample batches for learning. Breaks correlation between consecutive samples.

**Target Network**: Use a separate target network Q(s,a; θ⁻) for computing targets:
```
Target = r + γ max Q(s',a'; θ⁻)
```

Updates target network periodically by copying online network weights.

**Double DQN**: Reduces overestimation bias by using online network for action selection and target network for evaluation:
```
Target = r + γ Q_target(s', argmax Q_online(s',a'))
```

**Dueling Architecture**: Separates value V(s) and advantage A(s,a):
```
Q(s,a) = V(s) + (A(s,a) - mean A(s,a))
```

**Prioritized Experience Replay**: Sample transitions with probability proportional to TD-error magnitude, focusing learning on surprising experiences.

## Reward Shaping

### Sparse Rewards

- Win: +100
- Loss: -100
- Draw: 0

### Dense Rewards

Provide intermediate feedback:
- Goal entry: +100
- Capture: +50
- Safety: +15
- Boost: +10
- Progress: +1
- Death: -20

### Context-Aware Rewards

Scale rewards based on game context and move potential:
- **Trailing**: Boost aggressive moves, reduce conservative
- **Leading**: Boost defensive moves, reduce aggression
- **Neutral**: No scaling

This helps the agent adapt its strategy to the current game situation.

## State Representation

### Full State Vector

For neural networks, we use a normalized feature vector:
- Piece positions (normalized 0-1)
- Enemy piece positions
- Dice roll (normalized)
- Current player (one-hot)
- Turn indicator

### Abstract State

For tabular methods, we use a hashable tuple:
- (P1, P2, P3, P4, Context)
- P1-P4: Potential categories for each piece
- Context: Global game context (trailing/neutral/leading)

## Exploration vs Exploitation

### ε-Greedy Policy

Balance exploration and exploitation:
- With probability ε: random action (exploration)
- With probability 1-ε: greedy action argmax Q(s,a) (exploitation)

### Epsilon Decay

Start with high ε (exploration) and decay over time:
```
εₜ = max(ε_min, ε₀ × decay^t)
```

## Multi-Agent Considerations

Ludo is a multi-agent game with:
- **Stochastic Opponents**: Opponent actions are not fully predictable
- **Non-Stationarity**: Opponent strategies may change
- **Partial Observability**: Cannot directly observe opponent strategies

Our experiments use random opponents as a baseline. Results may differ against stronger or learning opponents.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*.

