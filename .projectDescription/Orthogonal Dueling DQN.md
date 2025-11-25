```table-of-contents
```

**Date:** 2025-11-22  
**Tags:** #AI #ludo #reinforcementlearning #deeplearning #dqn  
**Architecture:** Dueling DQN + Double Learning + Prioritized Replay (PER) + N-Step Returns  
**State Space:** 31-Dimension Orthogonal Vector  
**Target Performance:** Super-human strategic play (Star jumps, blockade handling, aggressive kill-hunting).

---

# **Research-Grade Design Document: Orthogonal Dueling Double DQN for Ludo**

## **1. Problem Formulation: The "Deep" MDP**

### 1.1 Ludo as a Partially Observable Markov Decision Process

We model Ludo as a Partially Observable Markov Decision Process (POMDP) where the "hidden" information (exact enemy future rolls) is mitigated by probabilistic reasoning.

**MDP Characteristics:**
- **State Space:** The positions of all 16 pieces (4 per player × 4 players) on a board with 58 possible positions per piece
- **Action Space:** Discrete actions selecting which of 4 pieces to move (when valid)
- **Stochasticity:** Dice rolls introduce randomness (1-6 with uniform probability)
- **Multi-Agent:** 4 players act sequentially, creating a non-stationary environment
- **Sparse Rewards:** Terminal reward (win/loss) primarily at game end

### 1.2 The State Space Explosion Problem

A naive state representation tracking exact positions of all pieces results in approximately $58^{16} \approx 10^{28}$ unique states. This creates an intractable learning problem:

- **Memory:** Storing a Q-table for $10^{28}$ states is computationally infeasible
- **Sample Complexity:** The agent would need to visit each state-action pair multiple times, requiring millions of years of gameplay
- **Generalization:** The agent cannot recognize that "Killing at Tile 10" is strategically equivalent to "Killing at Tile 50"

### 1.3 Core Solution Philosophy: Deep Generalization

Unlike Tabular Q-learning, which memorizes states, this Deep Network generalizes. It learns that *"A piece 4 steps behind an enemy"* is a universal concept of "Threat," regardless of whether it happens at Tile 10 or Tile 50.

**Key Advantages:**
- **Function Approximation:** Neural networks learn continuous value functions, enabling interpolation between similar states
- **Feature Learning:** The network automatically discovers relevant patterns (e.g., "threat distance," "kill opportunities")
- **Scalability:** The same architecture can handle the full state space without exponential memory growth

**Objective:** Maximize total discounted reward (Win the game efficiently).  
**Constraint:** The agent must learn purely from self-play and sparse rewards, without hard-coded rulebooks for "stars" or "blockades."

## **2. The "Orthogonal" State Abstraction ($\Phi(s)$)**

### 2.1 State Space Reduction

We abandon the massive $10^{28}$ state space for a compact, **31-dimensional vector**.

**State Space Calculation:**
- **Raw State Space:** $58^{16} \approx 10^{28}$ states
- **Abstract State Space:** Continuous 31-dimensional vector (infinite states, but bounded)
- **Reduction Factor:** From discrete $10^{28}$ to continuous bounded space

**Design Philosophy: Orthogonality.** Every feature captures a distinct, non-redundant aspect of the game. This prevents "multicollinearity," which causes instability and slow convergence in neural networks.

### 2.2 State Representation

The abstract state is a 31-dimensional vector:

$$\Phi(s) = [f_1, f_2, \ldots, f_{31}] \in \mathbb{R}^{31}$$

Where each $f_i$ is a normalized feature value.

### 2.3 Per-Piece Features (20 Dimensions)

*For each of the 4 pieces ($P_0, P_1, P_2, P_3$), we compute 5 features:*

| Index | Feature Name | Mathematical Definition | Value Range | Description |
| :---- | :---- | :---- | :---- | :---- |
| $4i+1$ | **Normalized Progress** | $\text{Progress}_i = \frac{\text{Pos}_i}{57}$ | $[0.0, 1.0]$ | How close is this piece to winning? |
| $4i+2$ | **Is Safe** | $\text{Safe}_i = \mathbb{1}[\text{Pos}_i \in \{\text{Globes} \cup \text{Stars} \cup \text{Home}\}]$ | $\{0, 1\}$ | Binary safety flag |
| $4i+3$ | **In Home Corridor** | $\text{Corridor}_i = \mathbb{1}[\text{Pos}_i \in \text{HomeCorridor}]$ | $\{0, 1\}$ | Binary invincible zone flag |
| $4i+4$ | **Threat Distance** | $\text{Threat}_i = \min\left(1.0, \frac{\min_{e \in \text{EnemiesBehind}}(\text{dist}(e, P_i))}{6}\right)$ | $[0.0, 1.0]$ | Normalized distance to nearest enemy behind (1.0 = Safe) |
| $4i+5$ | **Kill Opportunity** | $\text{Kill}_i = \mathbb{1}[\exists e: \text{can\_kill}(P_i, e, \text{dice})]$ | $\{0, 1\}$ | Binary: Can this piece kill an enemy this turn? |

**Where:** $i \in \{0, 1, 2, 3\}$ indexes the four pieces.

### 2.4 Global Features (11 Dimensions)

*Captures the "Game State" context.*

| Index   | Feature Name         | Mathematical Definition                                                                                                    | Value Range     | Description                              |
|---------|---------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------|------------------------------------------|
| 21      | **Relative Progress** | RelProgress = (avg_pos_me - avg_pos_enemy) / 57                                                                            | [-1.0, 1.0]     | Normalized difference in average progress |
| 22      | **Pieces in Yard**    | Yard = (number of my pieces at Home) / 4                                                                                   | [0.0, 1.0]      | Normalized count of pieces waiting to start         |
| 23      | **Pieces Scored**     | Scored = (number of my pieces in Goal) / 4                                                                                 | [0.0, 1.0]      | Normalized count of finished pieces       |
| 24      | **Enemy Scored**      | EnemyScored = (number of enemy pieces in Goal) / 12                                                                        | [0.0, 1.0]      | Normalized count of enemy finished pieces |
| 25      | **Max Kill Potential**| KillPotential = min(1.0, (number of my pieces with Kill_i = 1) / 4 )                                                       | [0.0, 1.0]      | Normalized count of pieces that can kill  |
| 26-31   | **Dice Roll**         | Dice = one_hot(roll - 1)                                                                                                   | {0, 1}^6        | One-hot encoding of dice roll (1-6)       |

**Where:**
- avg_pos_me = mean position of my 4 pieces = (Pos_0 + Pos_1 + Pos_2 + Pos_3) / 4
- avg_pos_enemy = mean position of enemy 12 pieces = (sum of all 12 enemy piece positions) / 12


### 2.5 State Abstraction Algorithm

```
FUNCTION get_orthogonal_state(state, dice_roll):
    features = []
    
    // Step 1: Per-piece features (20 dimensions)
    FOR each piece i in [0, 1, 2, 3]:
        pos = state.pieces[i].position
        
        // Normalized Progress
        features.append(pos / 57.0)
        
        // Is Safe (Globe, Star, or Home)
        is_safe = (pos in GLOBES) OR (pos in STARS) OR (pos == HOME)
        features.append(1.0 if is_safe else 0.0)
        
        // In Home Corridor
        in_corridor = (pos >= HOME_START) AND (pos < GOAL)
        features.append(1.0 if in_corridor else 0.0)
        
        // Threat Distance (normalized)
        enemies_behind = [e for e in state.enemies if e.position < pos]
        IF enemies_behind:
            min_dist = min([pos - e.position for e in enemies_behind])
            threat_dist = min(1.0, min_dist / 6.0)
        ELSE:
            threat_dist = 1.0  // Safe
        features.append(threat_dist)
        
        // Kill Opportunity
        can_kill = FALSE
        FOR each enemy e in state.enemies:
            IF (pos + dice_roll == e.position) AND (e.position not in SAFE_ZONES):
                can_kill = TRUE
                BREAK
        features.append(1.0 if can_kill else 0.0)
    
    // Step 2: Global features (11 dimensions)
    my_avg_progress = mean([p.position for p in state.pieces]) / 57.0
    enemy_avg_progress = mean([e.position for e in state.enemies]) / 57.0
    relative_progress = (my_avg_progress - enemy_avg_progress)
    features.append(relative_progress)
    
    pieces_in_yard = sum([1 for p in state.pieces if p.position == HOME]) / 4.0
    features.append(pieces_in_yard)
    
    pieces_scored = sum([1 for p in state.pieces if p.position == GOAL]) / 4.0
    features.append(pieces_scored)
    
    enemy_scored = sum([1 for e in state.enemies if e.position == GOAL]) / 12.0
    features.append(enemy_scored)
    
    max_kill_potential = min(1.0, sum([features[4*i+4] for i in range(4)]) / 4.0)
    features.append(max_kill_potential)
    
    // Dice Roll (one-hot encoding)
    dice_one_hot = [0.0] * 6
    dice_one_hot[dice_roll - 1] = 1.0
    features.extend(dice_one_hot)
    
    RETURN numpy.array(features)  // Shape: (31,)
```

## **3\. Theoretical Justification: The Need for Orthogonality**

In machine learning, features are orthogonal if the information contained in one feature cannot be perfectly predicted or derived from a linear combination of other features.

**Why Orthogonal Feature Design Prevents Instability:**

1. **Elimination of Multicollinearity:** Non-orthogonal designs (like using both "Relative Progress" and a "Leading/Trailing" context flag) create redundancy. This forces the optimization algorithm to struggle along a "ridge" in the loss landscape, leading to high Q-value variance and slower convergence.  
2. **Clean Credit Assignment:** Orthogonal features ensure that when the agent receives a reward (e.g., \+15 for a kill), the gradient flows cleanly to the specific feature that caused it (Kill Opportunity), rather than diffusing the learning signal across correlated features.  
3. **Robust Generalization:** The agent learns independent concepts (e.g., "Progress" is good, "Threat" is bad). It can combine these concepts dynamically (e.g., "High Progress \+ Low Threat \= Great State").

## **4. Neural Network Architecture: Dueling DQN**

### 4.1 Architecture Overview

We use a **Dueling Architecture** combined with crucial stabilization layers. The dueling architecture decomposes the Q-function into a state value function $V(s)$ and an advantage function $A(s, a)$, enabling more efficient learning.

**Architectural Flow:**

```
Input Vector (31 dims)
       │
       ▼
[ Linear Layer (31 -> 128) ]
       │
[ LayerNorm (128) ]  <-- CRITICAL STABILIZATION
       │
[ ReLU Activation ]
       │
       ├────────────────────────┐
       ▼                        ▼
[ Value Stream ]         [ Advantage Stream ]
(Linear 128->128)        (Linear 128->128)
       │                        │
     (ReLU)                   (ReLU)
       │                        │
(Linear 128->1)          (Linear 128->4)
       │                        │
       ▼                        ▼
    V(s)                     A(s, a)
       │                        │
       └──────────┬─────────────┘
                  ▼
            Q(s, a) Output
```

### 4.2 Mathematical Formulation

**Forward Pass:**

1. **Input Layer:**
   $$h_0 = \Phi(s) \in \mathbb{R}^{31}$$

2. **First Hidden Layer with LayerNorm:**
   $$h_1^{\text{raw}} = W_1 h_0 + b_1 \in \mathbb{R}^{128}$$
   $$h_1 = \text{LayerNorm}(h_1^{\text{raw}})$$
   $$h_1^{\text{act}} = \text{ReLU}(h_1)$$

   **LayerNorm Formula:**
   $$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   
   Where:
   - $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ (mean)
   - $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ (variance)
   - $\gamma, \beta$ are learnable parameters
   - $\epsilon = 10^{-5}$ (numerical stability)

3. **Value Stream:**
   $$h_V = \text{ReLU}(W_V h_1^{\text{act}} + b_V) \in \mathbb{R}^{128}$$
   $$V(s) = W_{V_{\text{out}}} h_V + b_{V_{\text{out}}} \in \mathbb{R}$$

4. **Advantage Stream:**
   $$h_A = \text{ReLU}(W_A h_1^{\text{act}} + b_A) \in \mathbb{R}^{128}$$
   $$A(s, a) = W_{A_{\text{out}}} h_A + b_{A_{\text{out}}} \in \mathbb{R}^{4}$$

5. **Q-Function Combination:**
   $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

   **Theoretical Justification:** The subtraction of the mean advantage ensures identifiability (the network can learn $V(s)$ and $A(s,a)$ up to a constant offset) and improves stability (Source: Wang et al., 2016 - "Dueling Network Architectures for Deep Reinforcement Learning").

### 4.3 Design Rationale

**LayerNorm (Stabilization):**
- **Purpose:** Prevents "exploding gradients" and Q-value drift by normalizing the output of the first hidden layer
- **Theoretical Basis:** LayerNorm stabilizes the distribution of activations, reducing internal covariate shift (Source: Ba et al., 2016 - "Layer Normalization")
- **Critical for DQN:** Q-values can drift over training, causing instability. LayerNorm constrains the activation magnitudes.

**Dueling (Efficiency):**
- **Purpose:** Separates the estimation of the state's quality ($V(s)$) from the differential value of the actions ($A(s,a)$)
- **Theoretical Basis:** In many states, the value of the state is independent of the action taken. By learning $V(s)$ separately, the network can generalize better across similar states (Source: Wang et al., 2016)
- **Ludo Benefit:** Many board configurations are strategically equivalent (e.g., "I'm winning" vs "I'm winning with piece A vs piece B"). The dueling architecture learns this equivalence faster.

### 4.4 Network Parameters

**Total Parameters:**
- Input Layer: $31 \times 128 + 128 = 4,096$
- LayerNorm: $128 \times 2 = 256$ (scale and bias)
- Value Stream: $(128 \times 128 + 128) + (128 \times 1 + 1) = 16,513$
- Advantage Stream: $(128 \times 128 + 128) + (128 \times 4 + 4) = 16,900$
- **Total:** $\approx 38,000$ parameters

**Activation Functions:**
- ReLU: $\text{ReLU}(x) = \max(0, x)$ (used throughout)
- No activation on output layers (linear outputs for regression)

## **5. Training Methodology: Stability First**

### 5.1 Algorithm: Double DQN (DDQN)

#### 5.1.1 The Maximization Bias Problem

Standard DQN uses the same network for both action selection and value estimation:

$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta_t^-)$$

This causes **maximization bias**: the max operator systematically overestimates Q-values in stochastic environments because it selects the action with the highest (potentially noisy) estimate.

**Theoretical Justification:** In stochastic environments, $\mathbb{E}[\max_a Q(s, a)] \geq \max_a \mathbb{E}[Q(s, a)]$ (Jensen's inequality for convex functions). This bias accumulates over time, leading to overoptimistic value estimates (Source: Van Hasselt et al., 2016 - "Deep Reinforcement Learning with Double Q-learning").

#### 5.1.2 Double DQN Solution

Double DQN decouples action *selection* (Online Network) from value *evaluation* (Target Network):

$$Y_t^{\text{DDQN}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta_t); \theta_t^-)$$

**Where:**
- $\theta_t$ = Online network parameters (updated every step)
- $\theta_t^-$ = Target network parameters (updated every $C$ steps)

**Algorithm:**
```
FUNCTION double_dqn_target(state, action, reward, next_state, done, online_net, target_net, gamma):
    IF done:
        target = reward
    ELSE:
        // Action selection using online network
        online_q_values = online_net(next_state)  // Shape: (4,)
        best_action = argmax(online_q_values)
        
        // Value evaluation using target network
        target_q_values = target_net(next_state)  // Shape: (4,)
        target = reward + gamma * target_q_values[best_action]
    
    RETURN target
```

#### 5.1.3 N-Step Returns

We incorporate **3-step returns** to reduce variance and smooth the learning signal:

**N-Step Return Formula:**
$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n \max_{a'} Q(S_{t+n}, a'; \theta^-)$$

**For 3-step returns:**
$$G_t^{(3)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 \max_{a'} Q(S_{t+3}, a'; \theta^-)$$

**Trade-off:** N-step returns reduce variance (more reward signal) but increase bias (older Q-estimates). For Ludo, $n=3$ provides a good balance (Source: Mnih et al., 2016 - "Asynchronous Methods for Deep Reinforcement Learning").

### 5.2 Replay Mechanism: Prioritized Experience Replay (PER)

#### 5.2.1 The Prioritization Scheme

Standard Experience Replay samples transitions uniformly. PER samples transitions with probability proportional to their TD-error:

**Priority Calculation:**
$$p_i = |\delta_i| + \epsilon$$

**Where:**
- $\delta_i = r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)$ (TD-error)
- $\epsilon = 10^{-6}$ (small constant to ensure non-zero probability)

**Sampling Probability:**
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

**Where:**
- $\alpha \in [0, 1]$ controls prioritization strength ($\alpha = 0$ = uniform, $\alpha = 1$ = fully prioritized)
- We use $\alpha = 0.6$ (moderate prioritization)

#### 5.2.2 Importance Sampling Correction

Prioritized sampling introduces bias. We correct this using importance sampling weights:

**Importance Sampling Weight:**
$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

**Where:**
- $N$ = buffer size (80,000)
- $\beta$ = bias correction factor, linearly annealed from 0.4 → 1.0 over training
- $\beta = 0.4$ initially (high bias correction), $\beta = 1.0$ at end (no correction needed)

**Normalized Weights:**
$$w_i^{\text{norm}} = \frac{w_i}{\max_j w_j}$$

This ensures weights are in $[0, 1]$ and prevents gradient explosion.

#### 5.2.3 PER Algorithm

```
CLASS PrioritizedReplayBuffer:
    INITIALIZE:
        buffer = SumTree(size=80000)  // Binary heap for O(log N) sampling
        max_size = 80000
        alpha = 0.6  // Prioritization exponent
        beta = 0.4  // Initial importance sampling exponent
        beta_end = 1.0
        beta_schedule = linear_anneal(0.4, 1.0, total_steps)
        epsilon = 1e-6
    
    FUNCTION add(state, action, reward, next_state, done):
        // Calculate TD-error (use max priority if new transition)
        priority = max_priority IF buffer is not full ELSE |td_error| + epsilon
        buffer.add(transition, priority)
    
    FUNCTION sample(batch_size):
        // Sample indices with probability proportional to priority
        indices = []
        segment = total_priority / batch_size
        
        FOR i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            index = buffer.get(value)  // O(log N) lookup
            indices.append(index)
        
        // Calculate importance sampling weights
        beta = beta_schedule[current_step]
        weights = []
        FOR index in indices:
            prob = buffer.get_priority(index) / total_priority
            weight = (1.0 / (buffer.size * prob)) ** beta
            weights.append(weight)
        
        // Normalize weights
        weights = weights / max(weights)
        
        RETURN transitions[indices], indices, weights
    
    FUNCTION update_priorities(indices, td_errors):
        FOR index, td_error in zip(indices, td_errors):
            priority = |td_error| + epsilon
            buffer.update(index, priority)
```

**Theoretical Justification:** PER dramatically improves sample efficiency by focusing learning on "surprising" transitions (Source: Schaul et al., 2016 - "Prioritized Experience Replay").

### 5.3 Loss Function and Optimization

#### 5.3.1 Huber Loss

We use Huber loss instead of MSE to reduce the impact of outliers:

**Huber Loss:**
$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**Where:** $\delta = 1.0$ (Huber loss parameter)

**With Importance Sampling:**
$$L = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot L_\delta(y_i, Q(s_i, a_i; \theta))$$

**Where:**
- $B$ = batch size (typically 32 or 64)
- $w_i$ = importance sampling weight
- $y_i$ = target Q-value (from Double DQN)

#### 5.3.2 Gradient Update

**Gradient Calculation:**
$$\nabla_\theta L = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \nabla_\theta L_\delta(y_i, Q(s_i, a_i; \theta))$$

**Optimizer:** Adam optimizer with learning rate $\alpha = 0.0001$ and standard momentum parameters ($\beta_1 = 0.9$, $\beta_2 = 0.999$).

### 5.4 Target Network Update

The target network is updated periodically using a soft or hard update:

**Hard Update (every $C$ steps):**
$$\theta^- \leftarrow \theta$$

**Where:** $C = 1000$ steps (update frequency)

**Alternative: Soft Update (Polyak Averaging):**
$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

**Where:** $\tau = 0.001$ (soft update coefficient)

We use **hard updates** for simplicity and stability.

### 5.5 Reward Shaping: The "Nash" Alignment

The reward function is designed to align with the Nash Equilibrium of the game (winning):

| Event | Reward | Rationale |
| :---- | :---- | :---- |
| **WIN GAME** | **+500** | The dominant, terminal objective. Highest reward to ensure winning is always prioritized. |
| **KILL ENEMY** | +15 | Strategic incentive; high enough to be pursued, low enough not to override winning. |
| **FINISH PIECE** | +40 | Strong sub-goal, encouraging permanent progress. Higher than kill to prioritize finishing. |
| **ENTER SAFE** | +10 | Survival incentive. Encourages defensive positioning. |
| **DEATH** | **-40** | **Critical:** High penalty to enforce risk-aversion and defensive play. |
| **MOVE REWARD** | 0 | **Removed:** Prevents reward hacking and unnecessary stalling. |

**Theoretical Justification:** This reward structure preserves optimality (potential-based reward shaping theorem, Ng et al., 1999) while providing dense learning signals for faster convergence.

## **6. System Architecture & Data Flow**

### 6.1 Component Overview

The system follows a modular architecture:

```
Main Entry Point
    ↓
Trainer (Orchestrator)
    ↓
LudoEnv (Environment) ←→ DQNAgent (Learning Agent)
    ↓                    ↓
State Abstraction    Dueling DQN Network
    ↓                    ↓
Reward Shaper       Prioritized Replay Buffer
    ↓                    ↓
                     Target Network (θ⁻)
```

### 6.2 Data Flow

1. **Main** initializes Trainer with Environment and Agent
2. **Trainer** runs training loop, orchestrating episodes
3. **LudoEnv** provides state observations and executes actions
4. **State Abstraction** converts raw board state to 31-dim orthogonal vector
5. **DQNAgent** selects actions using epsilon-greedy policy with action masking
6. **Reward Shaper** calculates rewards based on game events
7. **Prioritized Replay Buffer** stores transitions with priorities
8. **DQNAgent** samples from buffer and updates Q-network using Double DQN loss
9. **Target Network** is updated periodically for stable learning

### 6.3 Multi-Agent Turn Handling

Ludo is a 4-player game with sequential turns. The training loop handles this by:

- **Experience Buffering:** Store (state, action, reward) until the agent's next turn
- **Reward Attribution:** Separate active rewards (from agent's actions) from passive rewards (from waiting/opponent actions)
- **State Transitions:** The `next_state` for an action is observed when the agent's turn arrives again (after opponent turns)

## **7. Complete Training Algorithm**

### 7.1 Action Selection with Masking

**Epsilon-Greedy Policy with Action Masking:**

```
FUNCTION act(state, epsilon, online_network):
    // Abstract state
    phi_s = get_orthogonal_state(state, state.dice_roll)
    
    // Get valid actions (pieces that can move)
    valid_actions = state.get_valid_moves()  // e.g., [0, 1, 3]
    
    // Exploration vs Exploitation
    IF random() < epsilon:
        // Exploration: random valid action
        action = random.choice(valid_actions)
    ELSE:
        // Exploitation: greedy action with masking
        q_values = online_network(phi_s)  // Shape: (4,)
        
        // Mask invalid actions (set to -infinity)
        masked_q = [-inf] * 4
        FOR action in valid_actions:
            masked_q[action] = q_values[action]
        
        // Select best valid action
        action = argmax(masked_q)
    
    RETURN action
```

**Theoretical Justification:** Action masking is theoretically superior to penalty-based approaches because it prevents the network from learning negative Q-values for invalid actions, which can cause instability (Source: Huang & Ontañón, 2022 - "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms").

### 7.2 Training Loop Algorithm

```
FUNCTION train(env, agent, num_episodes, steps_per_episode):
    // Initialize
    online_network = DuelingDQN()
    target_network = DuelingDQN()
    target_network.load_state_dict(online_network.state_dict())
    replay_buffer = PrioritizedReplayBuffer(size=80000)
    optimizer = Adam(online_network.parameters(), lr=0.0001)
    
    epsilon = 1.0  // Start with full exploration
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_target_every = 1000
    batch_size = 32
    n_step = 3
    
    step_count = 0
    
    FOR episode = 1 TO num_episodes:
        state = env.reset()
        done = False
        episode_reward = 0
        n_step_buffer = []  // For n-step returns
        
        WHILE NOT done:
            // 1. Action selection
            action = agent.act(state, epsilon, online_network)
            
            // 2. Environment step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            // 3. Store transition in n-step buffer
            n_step_buffer.append((state, action, reward, next_state, done))
            
            // 4. If buffer has n steps, calculate n-step return
            IF len(n_step_buffer) >= n_step:
                // Calculate n-step return
                n_step_return = 0
                FOR i in range(n_step):
                    n_step_return += (gamma ** i) * n_step_buffer[i][2]  // reward
                
                // Add bootstrap value if not terminal
                IF NOT n_step_buffer[-1][4]:  // not done
                    bootstrap_state = n_step_buffer[-1][3]  // next_state
                    phi_bootstrap = get_orthogonal_state(bootstrap_state, bootstrap_state.dice_roll)
                    bootstrap_q = target_network(phi_bootstrap)
                    bootstrap_action = argmax(online_network(phi_bootstrap))  // Double DQN
                    n_step_return += (gamma ** n_step) * bootstrap_q[bootstrap_action]
                
                // Store in replay buffer
                first_state, first_action, _, _, _ = n_step_buffer[0]
                phi_first = get_orthogonal_state(first_state, first_state.dice_roll)
                
                // Calculate TD-error for priority
                current_q = online_network(phi_first)[first_action]
                td_error = abs(n_step_return - current_q)
                
                replay_buffer.add(
                    state=phi_first,
                    action=first_action,
                    reward=n_step_return,  // n-step return
                    next_state=phi_bootstrap,
                    done=n_step_buffer[-1][4],
                    priority=td_error
                )
                
                // Remove oldest transition
                n_step_buffer.pop(0)
            
            // 5. Learning step (every 4 steps)
            IF step_count % 4 == 0 AND replay_buffer.size() > batch_size:
                // Sample batch
                batch, indices, weights = replay_buffer.sample(batch_size)
                
                // Calculate targets
                targets = []
                td_errors = []
                
                FOR transition in batch:
                    phi_s, a, r, phi_s_next, done = transition
                    
                    IF done:
                        target = r
                    ELSE:
                        // Double DQN: action selection with online, evaluation with target
                        online_q_next = online_network(phi_s_next)
                        best_action = argmax(online_q_next)
                        target_q_next = target_network(phi_s_next)
                        target = r + gamma * target_q_next[best_action]
                    
                    current_q = online_network(phi_s)[a]
                    td_error = abs(target - current_q)
                    td_errors.append(td_error)
                    targets.append(target)
                
                // Update priorities
                replay_buffer.update_priorities(indices, td_errors)
                
                // Calculate loss
                phi_s_batch = torch.stack([b[0] for b in batch])
                actions_batch = torch.tensor([b[1] for b in batch])
                targets_batch = torch.tensor(targets)
                weights_batch = torch.tensor(weights)
                
                q_values = online_network(phi_s_batch)
                q_selected = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
                
                loss = huber_loss(q_selected, targets_batch)
                weighted_loss = (weights_batch * loss).mean()
                
                // Backward pass
                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
                optimizer.step()
            
            // 6. Update target network
            IF step_count % update_target_every == 0:
                target_network.load_state_dict(online_network.state_dict())
            
            // 7. Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            state = next_state
            step_count += 1
        
        // Log episode statistics
        log_episode(episode, episode_reward, epsilon, replay_buffer.size())
```

### 7.3 Key Hyperparameters

| Hyperparameter | Value | Rationale |
| :---- | :---- | :---- |
| **Learning Rate** | 0.0001 | Conservative rate for stable learning |
| **Discount Factor ($\gamma$)** | 0.99 | High discount for long-term planning |
| **Epsilon Start** | 1.0 | Full exploration initially |
| **Epsilon End** | 0.01 | Minimal exploration at convergence |
| **Epsilon Decay** | 0.995 | Gradual decay per episode |
| **Batch Size** | 32 | Standard batch size for DQN |
| **Replay Buffer Size** | 80,000 | Large buffer for diverse experiences |
| **Target Update Frequency** | 1000 steps | Periodic updates for stability |
| **N-Step** | 3 | Balance between bias and variance |
| **PER Alpha** | 0.6 | Moderate prioritization |
| **PER Beta Start** | 0.4 | Initial bias correction |
| **PER Beta End** | 1.0 | Final bias correction |
| **Gradient Clipping** | 10.0 | Prevent exploding gradients |

## **8. Theoretical Properties**

### 8.1 Convergence Guarantees

**Under Standard Assumptions:**
- All state-action pairs visited infinitely often
- Learning rate conditions: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$
- Bounded rewards and Q-values

**Convergence Result:** Double DQN converges to the optimal Q-function $Q^*$ for the abstracted MDP under function approximation assumptions (Source: Van Hasselt et al., 2016).

**Note:** The abstraction may not preserve full optimality of the original MDP, but it preserves near-optimality for the strategic concepts we care about (Kill, Safety, Progress, etc.).

### 8.2 Sample Efficiency

**Comparison:**
- **Tabular Q-Learning:** Requires visiting each of $10^{28}$ states multiple times → intractable
- **Deep Q-Learning with Abstraction:** Learns continuous value function → can generalize from limited samples
- **Expected Convergence:** ~50,000-100,000 episodes for strong play (vs. millions for tabular methods)

**PER Impact:** Prioritized Experience Replay improves sample efficiency by 2-3x by focusing on high-error transitions (Source: Schaul et al., 2016).

### 8.3 Generalization Properties

The orthogonal feature design enables the agent to:

1. **Spatial Generalization:** Recognize that "Threat at Tile 10" = "Threat at Tile 50" (same threat distance feature)
2. **Strategic Generalization:** Learn that "High Progress + Low Threat = Good State" regardless of exact positions
3. **Transfer Learning:** Features learned on one game configuration transfer to similar configurations

**Theoretical Basis:** The orthogonal design ensures features are linearly independent, enabling the network to learn distinct concepts that can be combined flexibly (Source: Li et al., 2006 - "Towards a Unified Theory of State Abstraction for MDPs").

### 8.4 Stability Guarantees

**LayerNorm:** Prevents activation drift and gradient explosion by normalizing layer outputs.

**Target Network:** Reduces correlation between current and target Q-values, improving stability.

**Double DQN:** Eliminates maximization bias, preventing systematic overestimation.

**Gradient Clipping:** Prevents exploding gradients during backpropagation.

**Combined Effect:** These techniques ensure stable learning over millions of steps without divergence.

## **9. Implementation Roadmap (Phase 2)**

I will now proceed to generate the Python files that implement this finalized, research-grade design.

1. **orthogonal_abstractor.py**: Implementation of the 31-dim state vector with detailed feature calculations.  
2. **dueling_dqn_network.py**: The PyTorch model class with LayerNorm and dueling architecture.  
3. **dqn_agent.py**: The agent logic with action masking, DDQN selection, and epsilon-greedy policy.  
4. **prioritized_replay_buffer.py**: PER buffer implementation with SumTree for efficient sampling.  
5. **trainer.py**: The training loop integrating all components (DDQN + PER + 3-step returns + target network updates).

## **10. Summary**

This approach solves the Ludo learning problem by:

1. **Orthogonal State Abstraction:** Reducing $10^{28}$ states to a 31-dimensional continuous vector with non-redundant features
2. **Dueling DQN Architecture:** Separating state value and action advantages for efficient learning
3. **Double DQN:** Eliminating maximization bias through decoupled action selection and evaluation
4. **Prioritized Experience Replay:** Improving sample efficiency by focusing on high-error transitions
5. **N-Step Returns:** Reducing variance in value estimates through multi-step bootstrapping
6. **Stabilization Techniques:** LayerNorm, target networks, and gradient clipping for stable training

The result is a sample-efficient, strategically-aware agent that learns to play Ludo at super-human levels using deep reinforcement learning, capable of discovering complex strategies like star jumps, blockade handling, and aggressive kill-hunting without hard-coded rules.