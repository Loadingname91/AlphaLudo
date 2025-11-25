```table-of-contents
```

**Date:** 2025-11-20
**Tags:** #AI #ludo #reinforcementlearning 

---

# Context-Aware Potential-Based Tabular Q-Learning for Ludo

## 1. Problem Formulation & Approach

### 1.1 Ludo as a Markov Decision Process

Ludo is a stochastic, multi-agent board game that can be modeled as a Markov Decision Process (MDP) with the following characteristics:

- **State Space:** The positions of all 16 pieces (4 per player × 4 players) on a board with 58 possible positions per piece
- **Action Space:** Discrete actions selecting which of 4 pieces to move (when valid)
- **Stochasticity:** Dice rolls introduce randomness (1-6 with uniform probability)
- **Multi-Agent:** 4 players act sequentially, creating a non-stationary environment
- **Sparse Rewards:** Terminal reward (win/loss) only at game end

### 1.2 The State Space Explosion Problem

A naive state representation tracking exact positions of all pieces results in approximately $58^{16} \approx 10^{28}$ unique states. This creates an intractable learning problem for tabular Q-learning:

- **Memory:** Storing a Q-table for $10^{28}$ states is computationally infeasible
- **Sample Complexity:** The agent would need to visit each state-action pair multiple times, requiring millions of years of gameplay
- **Generalization:** The agent cannot recognize that "Killing at Tile 10" is strategically equivalent to "Killing at Tile 50"

### 1.3 Core Solution Philosophy

Instead of tracking **positions**, we abstract **capabilities**. The agent learns to recognize strategic opportunities (e.g., "I can kill an enemy," "I can reach safety") rather than memorizing board configurations. This capability-based abstraction:

- Reduces state space from $10^{28}$ to $7,203$ states (manageable for tabular methods)
- Enables generalization across similar strategic situations
- Preserves the Markov property while dramatically improving sample efficiency

---

## 2. State Abstraction Strategy

### 2.1 State Representation

The abstract state is a 5-tuple of integers:

$$S = (F_{P1}, F_{P2}, F_{P3}, F_{P4}, C)$$

Where:
- $F_{Pi}$ represents the **Potential** (capability) of piece $i$ after simulating its move
- $C$ represents the **Context** (game phase: Leading/Neutral/Trailing)

### 2.2 Piece Potentials: Capability Classification

For each piece, we simulate its move with the current dice roll and classify the outcome into one of 7 tactical categories:

| ID | Name | Trigger Condition | Strategic Intent |
|---|---|---|---|
| 0 | **Null** | Piece cannot move (stuck at home, finished, or blocked) | Action masked |
| 1 | **Neutral** | Moves to empty tile, no special interaction | Default progression |
| 2 | **Risk** | Lands where enemy is 1-6 steps behind | Danger alert - escape needed |
| 3 | **Boost** | Lands on Star (jump) or gains significant distance | Speed advantage |
| 4 | **Safety** | Lands on Globe, Safe Star, or enters Home Corridor | Defense - bank progress |
| 5 | **Kill** | Lands on opponent's tile, capturing them | Aggression - teleport enemy to start |
| 6 | **Goal** | Enters final winning triangle | Victory condition |

**Key Insight:** By classifying outcomes rather than positions, the agent learns that "Killing at any location" has the same strategic value, enabling instant generalization.

### 2.3 Context Calculation: Game Phase Detection

The context $C$ indicates whether the agent is winning or losing, enabling adaptive strategy:

**Weighted Equity Score:**
$$\text{Score} = (\text{Goal} \times 100) + (\text{Home\_Corridor} \times 50) + (\text{Safe\_Globe} \times 10) + (\text{Raw\_Distance})$$

**Gap Calculation:**
$$\text{Gap} = \text{My\_Score} - \max(\text{Opponent\_Scores})$$

**Context States:**

| ID | Name | Condition | Strategic Behavior |
|---|---|---|---|
| 0 | **Trailing** | $\text{Gap} < -20$ | Panic mode: High risk tolerance, value kills/boosts |
| 1 | **Neutral** | $-20 \leq \text{Gap} \leq 20$ | Balanced race: Standard play |
| 2 | **Leading** | $\text{Gap} > 20$ | Lockdown mode: Zero risk, prioritize safety |

### 2.4 State Space Reduction

- **Raw State Space:** $58^{16} \approx 10^{28}$ states
- **Abstract State Space:** $7^4 \times 3 = 7,203$ states
- **Reduction Factor:** $\approx 10^{24}$ (fits in L1 cache for extreme speed)

### 2.5 State Abstraction Algorithm

```
FUNCTION get_abstract_state(state):
    // Step 1: Calculate global context
    my_score = weighted_equity_score(state.player_pieces)
    opponent_scores = [weighted_equity_score(enemy) for enemy in state.enemy_pieces]
    gap = my_score - max(opponent_scores)
    
    IF gap < -20:
        context = TRAILING
    ELSE IF gap > 20:
        context = LEADING
    ELSE:
        context = NEUTRAL
    
    // Step 2: Calculate potential for each piece
    potentials = []
    FOR each piece i in [P1, P2, P3, P4]:
        IF piece cannot move:
            potential = NULL
        ELSE:
            // Simulate move with current dice roll
            next_position = simulate_move(piece.position, dice_roll)
            
            // Classify outcome (priority order)
            IF next_position == GOAL:
                potential = GOAL
            ELSE IF can_capture_enemy(next_position):
                potential = KILL
            ELSE IF is_safe_zone(next_position):
                potential = SAFETY
            ELSE IF is_star_jump(next_position):
                potential = BOOST
            ELSE IF is_threatened(next_position):
                potential = RISK
            ELSE:
                potential = NEUTRAL
        
        potentials.append(potential)
    
    RETURN (potentials[0], potentials[1], potentials[2], potentials[3], context)
```

---

## 3. Dynamic Reward Shaping

### 3.1 Context-Aware Reward Scaling

To teach the agent different strategies for different game phases, we apply **dynamic reward multipliers** based on both the action's potential and the current context.

**Core Principle:** The same action (e.g., "Kill") has different strategic value depending on whether we're winning or losing.

### 3.2 Reward Multipliers by Context

| Event | Base Reward | Trailing Multiplier | Leading Multiplier | Rationale |
|---|---|---|---|---|
| **Goal** | +100 | ×1.0 (+100) | ×1.0 (+100) | Always optimal |
| **Kill** | +50 | **×1.5 (+75)** | ×0.8 (+40) | Critical when losing, less important when winning |
| **Safety** | +15 | ×0.5 (+7.5) | **×2.0 (+30)** | Low priority when losing, essential when winning |
| **Boost** | +10 | ×1.2 (+12) | ×1.0 (+10) | Speed matters in race |
| **Neutral** | +1 | ×1.0 (+1) | ×1.0 (+1) | Baseline progression |
| **Risk** | -10 | ×0.5 (-5) | **×2.0 (-20)** | Acceptable when losing, fatal when winning |
| **Death** | -20 | ×1.0 (-20) | ×1.0 (-20) | Always bad |

### 3.3 Reward Scaling Algorithm

```
FUNCTION scale_reward(base_reward, potential, context):
    multiplier = 1.0
    
    IF context == TRAILING:
        IF potential == KILL:
            multiplier = 1.5
        ELSE IF potential == BOOST:
            multiplier = 1.2
        ELSE IF potential == SAFETY:
            multiplier = 0.5
        ELSE IF potential == GOAL:
            multiplier = 0.5
        ELSE IF potential == RISK:
            multiplier = 0.5
    
    ELSE IF context == LEADING:
        IF potential == KILL:
            multiplier = 0.8
        ELSE IF potential == SAFETY:
            multiplier = 2.0
        ELSE IF potential == RISK:
            multiplier = 2.0
    
    // Neutral context: multiplier = 1.0 (default)
    
    RETURN base_reward × multiplier
```

**Theoretical Justification:** This reward shaping preserves optimality (potential-based reward shaping theorem) while dramatically accelerating learning by providing clear signals about which actions are valuable in which contexts.

---

## 4. System Architecture & Flow

### 4.1 Component Overview

The system follows a modular architecture:

```
Main Entry Point
    ↓
Trainer (Orchestrator)
    ↓
LudoEnv (Environment) ←→ QLearningAgent (Learning Agent)
    ↓                    ↓
State Abstraction    Q-Table
    ↓
Reward Shaper
```

### 4.2 Data Flow

1. **Main** initializes Trainer with Environment and Agent
2. **Trainer** runs training loop, orchestrating episodes
3. **LudoEnv** provides state observations and executes actions
4. **State Abstraction** converts raw board state to abstract tuple
5. **QLearningAgent** selects actions using epsilon-greedy policy
6. **Reward Shaper** calculates context-aware rewards
7. **QLearningAgent** updates Q-values using Bellman equation

### 4.3 Multi-Agent Turn Handling

Ludo is a 4-player game with sequential turns. The training loop handles this by:

- **Experience Buffering:** Store (state, action, reward) until the agent's next turn
- **Reward Decomposition:** Separate active rewards (from agent's actions) from passive rewards (from waiting/opponent actions)
- **State Transitions:** The `next_state` for an action is observed when the agent's turn arrives again (after opponent turns)

---

## 5. Training Algorithm

### 5.1 Off-Policy Training Loop

The trainer implements an off-policy learning loop suitable for Q-learning:

```
FUNCTION run_off_policy_training(env, agent, num_episodes):
    FOR episode = 1 TO num_episodes:
        state = env.reset()
        pending_experience = None
        done = False
        
        WHILE NOT done:
            IF current_player == learning_agent_id:
                // Process pending experience from previous turn
                IF pending_experience != None:
                    agent.push_to_replay_buffer(
                        pending_experience.state,
                        pending_experience.action,
                        pending_experience.reward,
                        current_state,  // This is next_state for previous action
                        done = False
                    )
                    agent.learn_from_replay()
                    pending_experience = None
                
                // Select action using epsilon-greedy
                action = agent.act(state)
                
                // Execute action
                next_state, total_reward, done, info = env.step(action)
                
                // Decompose reward
                active_reward = info.active_reward  // Reward for this action
                passive_reward = info.passive_reward  // Penalty for waiting
                
                // Add passive reward to previous experience (if exists)
                IF previous_pending_experience != None:
                    previous_pending_experience.reward += passive_reward
                    agent.push_to_replay_buffer(...)
                    agent.learn_from_replay()
                
                // Create new pending experience
                pending_experience = {
                    state: current_state,
                    action: action,
                    reward: active_reward
                }
                
                state = next_state
                
            ELSE:
                // Opponent's turn - environment handles automatically
                next_state, reward, done, info = env.step(opponent_action)
                state = next_state
        
        // End of episode: push final experience
        IF pending_experience != None:
            agent.push_to_replay_buffer(
                pending_experience.state,
                pending_experience.action,
                pending_experience.reward,
                terminal_state,
                done = True
            )
            agent.learn_from_replay()
        
        // Decay epsilon
        agent.on_episode_end()
```

### 5.2 Key Design Decisions

- **Experience Buffering:** Necessary because opponent turns intervene between agent action and next agent observation
- **Reward Decomposition:** Ensures rewards are correctly attributed to the actions that caused them
- **Episode Termination:** Final experience is marked as terminal (no future rewards)

---

## 6. Q-Learning Agent Algorithm

### 6.1 Q-Table Structure

The Q-table maps abstract states to action values:

```
Q-table: Dict[State_Tuple → Array[4]]
    Key: (F_P1, F_P2, F_P3, F_P4, C)  // 5-tuple
    Value: [Q(s, a_0), Q(s, a_1), Q(s, a_2), Q(s, a_3)]  // Q-values for 4 pieces
```

**Initialization:** All Q-values start at 0 (optimistic initialization could be used but not required).

### 6.2 Epsilon-Greedy Policy

```
FUNCTION act(state):
    // Abstract state
    state_tuple = state_abstractor.get_abstract_state(state)
    
    // Get valid actions (pieces that can move)
    valid_moves = state.valid_moves  // e.g., [0, 1, 3] means pieces 0, 1, 3 can move
    
    // Exploration vs Exploitation
    IF random() < epsilon:
        // Exploration: random valid action
        action = random.choice(valid_moves)
    ELSE:
        // Exploitation: greedy action
        q_values = q_table[state_tuple]
        
        best_action = None
        best_q = -infinity
        
        FOR each action_index in valid_moves:
            piece_index = state.movable_pieces[action_index]
            q_value = q_values[piece_index]
            
            IF q_value > best_q:
                best_q = q_value
                best_action = action_index
        
        action = best_action
    
    // Store for learning update
    last_state_tuple = state_tuple
    last_action = action
    last_piece_index = state.movable_pieces[action]
    
    RETURN action
```

### 6.3 Bellman Update with Dynamic Reward Scaling

```
FUNCTION push_to_replay_buffer(state, action, base_reward, next_state, done):
    // Extract context from state we acted in
    context = last_state_tuple[4]  // Context is 5th element
    
    // Extract potential of action we took
    piece_index = last_piece_index
    action_potential = last_state_tuple[piece_index]
    
    // Apply dynamic reward scaling
    scaled_reward = scale_reward(base_reward, action_potential, context)
    
    // Abstract next state
    next_state_tuple = state_abstractor.get_abstract_state(next_state)
    
    // Calculate target Q-value
    IF done:
        target_q = scaled_reward  // Terminal state: no future rewards
    ELSE:
        next_q_values = q_table[next_state_tuple]
        max_next_q = max(next_q_values)  // Standard Q-learning: max over actions
        target_q = scaled_reward + gamma × max_next_q
    
    // Current Q-value
    current_q = q_table[last_state_tuple][piece_index]
    
    // Bellman update
    new_q = current_q + alpha × (target_q - current_q)
    
    // Update Q-table
    q_table[last_state_tuple][piece_index] = new_q
```

**Bellman Equation (with reward scaling):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{\text{scaled}}(s, a, c) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha$ = learning rate
- $\gamma$ = discount factor
- $R_{\text{scaled}}(s, a, c)$ = context-aware scaled reward
- $c$ = context (Trailing/Neutral/Leading)

### 6.4 Epsilon Decay

```
FUNCTION on_episode_end():
    epsilon = max(min_epsilon, epsilon × epsilon_decay)
```

Epsilon decays once per episode (not per step) to gradually shift from exploration to exploitation.

---

## 7. Complete Algorithm Flow

### 7.1 End-to-End Execution

```
// ===== INITIALIZATION =====
main():
    env = LudoEnv(opponent_agents=[RandomAgent, RandomAgent, RandomAgent])
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        epsilon_decay=0.9995
    )
    trainer = Trainer(env, agent, config)
    trainer.run()

// ===== TRAINING LOOP =====
trainer.run():
    FOR episode = 1 TO num_episodes:
        state = env.reset()
        pending_experience = None
        
        WHILE NOT done:
            IF agent's turn:
                // 1. Process previous experience
                IF pending_experience:
                    agent.push_to_replay_buffer(...)
                    agent.learn_from_replay()
                
                // 2. State abstraction
                state_tuple = state_abstractor.get_abstract_state(state)
                
                // 3. Action selection (epsilon-greedy)
                action = agent.act(state)
                
                // 4. Environment step
                next_state, base_reward, done, info = env.step(action)
                
                // 5. Reward calculation (with dynamic scaling)
                context = state_tuple[4]
                potential = state_tuple[action_piece_index]
                scaled_reward = scale_reward(base_reward, potential, context)
                
                // 6. Store experience (will be processed on next turn)
                pending_experience = {
                    state: state,
                    action: action,
                    reward: scaled_reward
                }
                
                state = next_state
            
            ELSE:
                // Opponent turn
                next_state, _, done, _ = env.step(opponent_action)
                state = next_state
        
        // 7. Episode end: final Q-update
        IF pending_experience:
            agent.push_to_replay_buffer(..., done=True)
            agent.learn_from_replay()
        
        // 8. Epsilon decay
        agent.on_episode_end()

// ===== Q-LEARNING UPDATE =====
agent.push_to_replay_buffer(state, action, reward, next_state, done):
    // Extract context and potential
    context = last_state_tuple[4]
    potential = last_state_tuple[piece_index]
    
    // Scale reward
    scaled_reward = scale_reward(reward, potential, context)
    
    // Abstract next state
    next_state_tuple = state_abstractor.get_abstract_state(next_state)
    
    // Bellman update
    target = scaled_reward + gamma × max(q_table[next_state_tuple])
    q_table[last_state_tuple][piece_index] += alpha × (target - q_table[last_state_tuple][piece_index])
```

### 7.2 Learning Dynamics

1. **Early Training:** High epsilon → exploration → agent discovers various state-action pairs
2. **Mid Training:** Epsilon decays → exploitation increases → agent refines Q-values
3. **Late Training:** Low epsilon → mostly greedy → agent follows learned policy

The context-aware reward scaling ensures the agent learns different strategies for different game phases, creating a more robust and adaptive policy.

---

## 8. Theoretical Properties

### 8.1 Convergence Guarantees

Under standard Q-learning assumptions (all state-action pairs visited infinitely often, learning rate conditions), the algorithm converges to the optimal Q-function $Q^*$ for the abstracted MDP.

**Note:** The abstraction may not preserve full optimality of the original MDP, but it preserves near-optimality for the strategic concepts we care about (Kill, Safety, etc.).

### 8.2 Sample Efficiency

- **Without Abstraction:** $\approx 10^{28}$ states → intractable
- **With Abstraction:** $7,203$ states → feasible (convergence in ~10,000-15,000 episodes)

### 8.3 Generalization

The capability-based abstraction enables the agent to:
- Recognize that "Killing at Tile 10" = "Killing at Tile 50" (same potential)
- Transfer knowledge across similar strategic situations
- Adapt strategy based on game phase (context-aware)

---

## 9. Summary

This approach solves the Ludo learning problem by:

1. **State Abstraction:** Reducing $10^{28}$ states to $7,203$ via capability-based representation
2. **Dynamic Rewards:** Teaching context-appropriate strategies through adaptive reward scaling
3. **Tabular Q-Learning:** Standard off-policy algorithm with Bellman updates
4. **Multi-Agent Handling:** Experience buffering to handle sequential opponent turns

The result is a sample-efficient, strategically-aware agent that learns to play Ludo effectively using tabular methods without requiring neural networks or function approximation.
