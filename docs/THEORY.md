# Theoretical Foundations

This document outlines the core theoretical concepts of Reinforcement Learning (RL) used in this project. For specific implementation details of each agent (e.g., exact state tuples, network architectures), please refer to the [Agent Documentation](AGENTS.md).

## 1. The Environment as an MDP

We model the game of Ludo as a **Markov Decision Process (MDP)**, defined by the tuple $(S, A, P, R, \gamma)$:

- **State Space ($S$)**: The set of all possible board configurations. In Ludo, this includes the positions of all 16 pieces (4 per player) and the current dice roll. The raw state space size is approx. $57^{16} \approx 10^{28}$, which necessitates approximation methods.
- **Action Space ($A$)**: The set of valid moves. For a given dice roll, a player has at most 4 legal moves (one for each piece), though fewer if pieces are stuck or blocked.
- **Transition Function ($P$)**: $P(s'|s, a)$ is the probability of reaching state $s'$ after taking action $a$. In Ludo, this is stochastic due to the dice roll for the next turn and the moves of the 3 opponents.
- **Reward Function ($R$)**: $R(s, a, s')$ defines the immediate feedback. We use **Reward Shaping** to provide dense feedback (e.g., for capturing, entering safe zones) rather than just a sparse win/loss signal.
- **Discount Factor ($\gamma$)**: Determines the importance of future rewards. We typically use $\gamma \in [0.9, 0.99]$.

## 2. Value-Based Reinforcement Learning

Our agents rely on estimating the **Value Function** to make decisions.

### The Bellman Equation
The optimal action-value function $Q^*(s, a)$ obeys the Bellman Optimality Equation:

$$ Q^*(s, a) = \mathbb{E}_{s'} [R + \gamma \max_{a'} Q^*(s', a') | s, a] $$

This recursive relationship allows us to learn $Q$ values iteratively.

## 3. Algorithms Implemented

### Tabular Q-Learning
For the **Q-Learning Agent**, we solve the Bellman equation directly using a lookup table.
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$

**Key Challenge**: The raw state space is too large for a table.
**Solution**: **State Abstraction**. We map the raw board state to a compact feature tuple (e.g., "Is piece 1 safe?", "Can piece 2 kill?").
> See [Q-Learning Theory](agents/q_learning/THEORY.md) for details on the abstraction logic.

### Deep Q-Networks (DQN)
For the **DQN Agent**, we use a neural network $Q(s, a; \theta)$ to approximate the Q-values.
$$ \text{Loss}(\theta) = \mathbb{E} [(y - Q(s, a; \theta))^2] $$
$$ y = R + \gamma \max_{a'} Q(s', a'; \theta^-) $$

**Key Techniques Used**:
1.  **Experience Replay**: Storing transitions to break correlation and improve sample efficiency.
2.  **Target Networks**: Using a slowly updating network ($\theta^-$) to calculate targets ($y$) for stability.
3.  **Double DQN**: Decoupling action selection from evaluation to reduce overestimation bias.
4.  **Dueling Architecture**: Separating the estimation of State Value $V(s)$ and Action Advantage $A(s, a)$.
> See [DQN Theory](agents/dqn/THEORY.md) for network architecture and training specifics.

## 4. Reward Shaping
To accelerate learning, we augment the sparse "Win (+100) / Loss (-100)" signal with heuristic rewards:
- **Progress**: Small positive reward for moving forward.
- **Capture (Kill)**: Large positive reward for sending an opponent home.
- **Safety**: Reward for entering a Globe or Star.
- **Goal**: Large reward for finishing a piece.

We also implement **Context-Aware Shaping**, where rewards are scaled dynamically based on whether the agent is winning or losing.

## 5. Multi-Agent Dynamics
Ludo is a 4-player zero-sum (or constant-sum) game.
- **Non-Stationarity**: From the perspective of one agent, the opponents are part of the environment. If opponents learn, the environment changes.
- **Self-Play**: Ideally, agents should train against themselves to reach Nash Equilibrium. Currently, we train against a mix of Random and Rule-Based opponents.

## References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Van Hasselt, H., et al. (2016). *Deep Reinforcement Learning with Double Q-Learning*. AAAI.
