# Tabular Q-Learning Agent Methodology

## Theory

The Tabular Q-Learning Agent uses context-aware state abstraction and dynamic reward scaling based on game context (trailing, neutral, leading) and move potentials (kill, boost, safety, risk, goal, neutral, null).

## State Abstraction

The agent supports two state abstraction modes, selectable via the `state_abstraction` parameter:

### Mode 1: Potential-Based (Default)

**State Tuple**: `(P1, P2, P3, P4, Context)`

- **P1-P4**: Potential classification for each piece (0-6)
- **Context**: Game context (0=Trailing, 1=Neutral, 2=Leading)

### Mode 2: Zone-Based (Inspired by [Ludo_Game_AI](https://github.com/raffaele-aurucci/Ludo_Game_AI))

**State Tuple**: `(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)`

- **HOME**: Count of tokens in base (0-4)
- **PATH**: Count of tokens on main path (0-4)
- **SAFE**: Count of tokens in safe zones (0-4)
- **GOAL**: Count of tokens that reached goal (0-4)
- **EV1-EV4**: Enemy vulnerable to agent's token (0-1, boolean)
- **TV1-TV4**: Agent's token under attack (0-1, boolean)

**Advantages of Zone-Based Abstraction**:
- More abstract representation reduces state space complexity
- Focuses on token distribution rather than individual positions
- Direct tactical information (vulnerabilities and threats) per token
- Better generalization across similar game states

### Potential Classifications

| Potential | Value | Description |
|-----------|-------|-------------|
| POT_NULL | 0 | Piece cannot move (home without dice=6, goal, not in movable_pieces) |
| POT_NEUTRAL | 1 | Normal move with no special outcome |
| POT_RISK | 2 | Move lands in threatened position (enemy 1-6 steps behind) |
| POT_BOOST | 3 | Star jump (extra movement via star tile) |
| POT_SAFETY | 4 | Move lands in safe zone (globe, home corridor, start) |
| POT_KILL | 5 | Move captures an enemy piece |
| POT_GOAL | 6 | Move reaches goal (winning move) |

### Context Calculation

Context is determined by **Weighted Equity** scoring:

**Weighted Score Formula**:
```
Score = (Goal × 100) + (Corridor × 50) + (Safe × 10) + (Distance)
```

**Scoring Details**:
- Goal position (57): +100 points
- Home Corridor (52-56): +50 + position
- Safe Globes/Start (1, 9, 22, 35, 48): +10 + position
- Main board: +position (raw distance)

**Context Thresholds**:
- **Trailing** (0): `my_score - max_opponent_score < -20`
- **Neutral** (1): `-20 <= gap <= 20`
- **Leading** (2): `gap > 20`

## Q-Table Structure

- **Key**: State tuple (varies by abstraction mode)
  - Potential mode: `(P1, P2, P3, P4, Context)`
  - Zone-based mode: `(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)`
- **Value**: Array of 4 Q-values (one per piece: Q[piece_idx])
- **Initialization**: `defaultdict(lambda: np.zeros(4))` (unseen states default to zeros)

## Action Selection

### Epsilon-Greedy Policy

```python
if random.random() < epsilon:
    action = random.choice(valid_moves)  # Exploration
else:
    # Exploitation: select piece with highest Q-value
    best_action = argmax(Q[state_tuple][piece_idx] for piece_idx in valid_moves)
```

### Action Index Mapping

- Action index → Piece index mapping:
  - If `state.movable_pieces` exists: `piece_idx = state.movable_pieces[action_idx]`
  - Else: `piece_idx = action_idx` (direct mapping)

## Learning: Bellman Update

### Standard Q-Learning Update

```
Q(s,a) = Q(s,a) + α[r_scaled + γ × max Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate (default: 0.1)
- `γ` (gamma): Discount factor (default: 0.9)
- `r_scaled`: Context-aware scaled reward
- `s'`: Next state (after action)

### Context-Aware Reward Scaling

Rewards are scaled based on context and action potential:

**Trailing Context** (Panic Mode):
- Kill: ×1.5 (desperate for captures)
- Boost: ×1.2 (need speed)
- Safety: ×0.8 (hiding won't help)
- Goal: ×0.5 (unlikely, but still valuable)

**Leading Context** (Lockdown Mode):
- Kill: ×0.8 (don't chase kills if it exposes us)
- Safety: ×2.0 (defense is paramount)
- Risk: ×1.5 (massive penalty for danger)

**Neutral Context**:
- Multiplier: ×1.0 (standard play)

## Edge Cases

### Unmovable Pieces
- **Home without dice=6**: POT_NULL (cannot exit)
- **Goal position**: POT_NULL (already finished)
- **Not in movable_pieces**: POT_NULL (ludopy says cannot move)
- **Special case**: If ludopy says movable but piece is at home with dice≠6, manually analyze (treat as if exiting)

### First Action
- **Scenario**: `last_state_tuple is None` (first action of episode)
- **Handling**: Skip Q-update (return early from `push_to_replay_buffer`)
- **Rationale**: No previous state to update from

### Invalid Action Indices
- **Scenario**: Action index out of bounds or doesn't map to valid piece
- **Handling**: 
  - Check `state.movable_pieces` exists and has enough elements
  - Fallback to direct mapping: `piece_idx = action_idx`
- **Rationale**: Handle different state representations

### Empty Enemy Pieces
- **Scenario**: `state.enemy_pieces` is empty or all enemies at home/goal
- **Handling**: 
  - Context calculation: `max_opponent_score = 0` if empty
  - Potential classification: No kill opportunities, no threats
- **Rationale**: Handle edge cases in multi-player games

### Potential Classification Edge Cases

#### Home with dice=6
- **Scenario**: Piece at home (0), dice roll is 6
- **Handling**: 
  - If in `movable_pieces`: Manually calculate `next_pos = 0 + 6 = 6`
  - Check if goal (6 == 57? No) or safe (6 in SAFE_GLOBES? Check)
  - Classify as POT_SAFETY or POT_NEUTRAL
- **Rationale**: Special exit case, cannot use normal simulate_move

#### Goal Bounce
- **Scenario**: Move overshoots goal (position > 57)
- **Handling**: `simulate_move` handles bounce: `GOAL_INDEX - overshoot`
- **Rationale**: Already handled in board physics

#### Star Jump Chains
- **Scenario**: Multiple consecutive star jumps (shouldn't happen, but document)
- **Handling**: `simulate_move` handles single jump, chains are prevented by game rules
- **Rationale**: Edge case documentation for completeness

### Q-Table Initialization
- **Scenario**: Unseen state tuple
- **Handling**: `defaultdict(lambda: np.zeros(4))` returns zeros for unseen states
- **Rationale**: Optimistic initialization (zeros), agent learns from experience

### Context Threshold Boundaries
- **Scenario**: Gap exactly -20 or +20
- **Handling**: 
  - `gap < -20` → Trailing (excludes -20, so -20 is Neutral)
  - `gap > 20` → Leading (excludes +20, so +20 is Neutral)
- **Rationale**: Strict boundaries prevent oscillation at threshold

### Reward Scaling Edge Cases
- **Unknown Potential**: If potential not in map, multiplier = 1.0 (no scaling)
- **Unknown Context**: If context not Trailing/Leading, multiplier = 1.0 (neutral)
- **Rationale**: Graceful degradation for edge cases

### Epsilon Decay
- **Schedule**: Per episode (not per step)
- **Formula**: `epsilon = max(min_epsilon, epsilon × epsilon_decay)`
- **Default**: Start 0.1, decay 0.9995, min 0.01
- **Rationale**: Slow decay maintains exploration throughout training

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate (α) | 0.1 | How quickly agent learns (0.0-1.0) |
| discount_factor (γ) | 0.9 | Importance of future rewards (0.0-1.0) |
| epsilon | 0.1 | Initial exploration rate |
| epsilon_decay | 0.9995 | Epsilon decay per episode |
| min_epsilon | 0.01 | Minimum exploration rate |
| state_abstraction | 'potential' | State abstraction mode: 'potential' or 'zone_based' |

## Implementation Checklist

- [ ] Initialize Q-table as `defaultdict(lambda: np.zeros(4))`
- [ ] Initialize state abstractor with `player_id`
- [ ] Set up reward scaling multipliers (trailing, leading, neutral)
- [ ] Implement epsilon-greedy action selection
- [ ] Map action indices to piece indices correctly
- [ ] Store `last_state_tuple`, `last_action`, `last_piece_idx` for Q-update
- [ ] Implement Bellman update with scaled rewards
- [ ] Handle first action edge case (skip update if `last_state_tuple is None`)
- [ ] Implement epsilon decay per episode (`on_episode_end`)
- [ ] Implement save/load for Q-table (pickle)
- [ ] Handle all potential classification edge cases
- [ ] Set `is_on_policy = False` and `needs_replay_learning = True`
- [ ] Implement optional score debugging (`supports_score_debug`, `get_last_score_debug`)

