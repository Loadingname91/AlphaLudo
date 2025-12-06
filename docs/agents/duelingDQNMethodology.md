# Dueling DQN Agent Methodology

## Theory

The Dueling DQN Agent extends standard DQN with:
1. **Dueling Architecture**: Separates state value V(s) and action advantages A(s,a)
2. **Double Q-Learning**: Uses online network to select actions, target network to evaluate
3. **Prioritized Experience Replay (PER)**: Samples transitions with probability proportional to TD-error

## Dueling Architecture

### Network Structure

```
Input (state_dim) → Shared Layers → Split
                                    ├→ Value Stream → V(s)
                                    └→ Advantage Stream → A(s,a)
                                    
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

### Architecture Details

**Shared Layers**:
- `Linear(input_dim, hidden_dim)`
- `LayerNorm(hidden_dim)`
- `ReLU`

**Value Stream**:
- `Linear(hidden_dim, hidden_dim)` → `ReLU` → `Linear(hidden_dim, 1)`
- Outputs: Single scalar V(s) representing state value

**Advantage Stream**:
- `Linear(hidden_dim, hidden_dim)` → `ReLU` → `Linear(hidden_dim, num_actions)`
- Outputs: 4 values A(s,a) representing advantage for each piece

### Q-Value Combination

```python
q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
```

**Why subtract mean?**: Ensures identifiability. You cannot add a constant to both V(s) and A(s,a) without changing Q(s,a). This makes the decomposition unique.

### Batch Dimension Handling

- Supports both 1D input `(state_dim,)` and 2D input `(batch_size, state_dim)`
- If 1D: Unsqueeze to `(1, state_dim)`, process, squeeze back to `(num_actions,)`
- If 2D: Process directly, output `(batch_size, num_actions)`

## Double Q-Learning

### Standard DQN Problem

Standard DQN uses same network for action selection and evaluation:
```
target = r + γ × max Q_target(s', a')
```

This can overestimate Q-values because it uses max over noisy estimates.

### Double DQN Solution

Use online network to select action, target network to evaluate:
```python
# Select action with online network
next_q_online = online_net(next_states)
next_actions = next_q_online.argmax(1)

# Evaluate with target network
next_q_target = target_net(next_states)
next_q_selected = next_q_target.gather(1, next_actions.unsqueeze(1))
```

**Formula**:
```
target = r + γ × Q_target(s', argmax Q_online(s', a'))
```

This reduces overestimation bias.

## Prioritized Experience Replay (PER)

### Priority Calculation

Priority is based on TD-error:
```
priority = |TD_error|^α + ε
```

Where:
- `α` (alpha): Prioritization exponent (0=uniform, 1=full priority, default: 0.6)
- `ε` (epsilon): Small constant to ensure non-zero priority (default: 1e-6)

### Sampling

Transitions are sampled with probability proportional to priority:
```
P(i) = priority_i^α / Σ(priority_j^α)
```

Implementation uses SumTree for efficient sampling.

### Importance Sampling Weights

To correct for bias introduced by non-uniform sampling:
```
weight_i = (N × P(i))^(-β) / max_weight
```

Where:
- `N`: Buffer capacity
- `β` (beta): Importance sampling exponent (linearly annealed from 0.4 to 1.0)
- Normalized by max weight to prevent extreme values

### Beta Annealing

Beta is linearly annealed over training:
```python
beta = beta_start + (beta_end - beta_start) * (step / 100000)
beta = min(beta, beta_end)
```

Starts at 0.4 (more correction), ends at 1.0 (full correction).

## Learning: Bellman Update

### Dueling Double DQN with PER

```python
# Current Q-values (from online network)
q_values = online_net(states)
q_selected = q_values.gather(1, actions.unsqueeze(1))

# Double DQN: select with online, evaluate with target
next_q_online = online_net(next_states)
next_actions = next_q_online.argmax(1)
next_q_target = target_net(next_states)
next_q_selected = next_q_target.gather(1, next_actions.unsqueeze(1))

# Targets
targets = rewards + (gamma * next_q_selected * ~dones)

# TD-errors for priority updates
td_errors = |targets - q_selected|

# Loss with importance sampling weights
loss = smooth_l1_loss(q_selected, targets, reduction='none')
weighted_loss = (weights * loss).mean()
```

## Edge Cases

### All DQN Edge Cases
See `dqnAgentMethodology.md` for:
- Buffer size < batch_size
- Invalid action indices
- Empty movable pieces
- Train frequency
- Target network update
- Device handling
- Gradient clipping
- State representation mismatch
- Network initialization
- Terminal states
- Epsilon decay

### PER-Specific Edge Cases

#### Empty Buffer
- **Scenario**: Buffer is empty (no transitions yet)
- **Handling**: `buffer.size` returns 0, skip learning until `size >= batch_size`
- **Rationale**: Cannot sample from empty buffer

#### Max Priority Initialization
- **Scenario**: New transitions added to buffer
- **Handling**: Use `max_priority = 1.0` for new transitions
- **Rationale**: Give new transitions high priority to ensure they're sampled

#### Beta Annealing
- **Scenario**: Beta scheduling over training
- **Handling**: Linear interpolation from `beta_start` to `beta_end` over 100,000 steps
- **Edge Case**: Beta capped at `beta_end` (never exceeds 1.0)
- **Rationale**: Gradually increase importance sampling correction

#### SumTree Edge Cases
- **Capacity Overflow**: When buffer is full, oldest transitions are overwritten
- **Priority Updates**: Update priorities after learning, track `max_priority`
- **Rationale**: SumTree maintains efficient priority sampling

#### Importance Sampling Weights
- **Division by Zero**: Prevented by normalizing by `max_weight`
- **Weight Normalization**: `weights = weights / weights.max()`
- **Rationale**: Prevent extreme weights that could destabilize learning

### TD-Error Calculation Edge Cases

#### Terminal States
- **Scenario**: `done = True` (episode ended)
- **Handling**: `targets = rewards + (gamma * next_q * ~dones)`
- **Rationale**: No future reward if episode ended (multiply by `~dones`)

#### TD-Error Clipping
- **Scenario**: Extreme TD-errors
- **Handling**: Use absolute value: `td_errors = |targets - q_selected|`
- **Rationale**: Priority should be positive, magnitude matters

### Priority Updates Edge Cases

#### Epsilon Addition
- **Scenario**: TD-error could be zero
- **Handling**: `priority = |td_error| + epsilon`
- **Rationale**: Prevent zero priority (would never be sampled)

#### Max Priority Tracking
- **Scenario**: New max priority found
- **Handling**: `max_priority = max(max_priority, new_priority)`
- **Rationale**: Update max for future new transitions

### Dueling Architecture Edge Cases

#### Advantage Mean Subtraction
- **Scenario**: Ensuring identifiability
- **Handling**: `q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))`
- **Rationale**: Prevents adding constant to both V and A without changing Q

#### Batch Dimension Handling
- **Scenario**: 1D vs 2D input tensors
- **Handling**: 
  - Check `state.dim() == 1`
  - If 1D: `unsqueeze(0)` before processing, `squeeze(0)` after
  - If 2D: Process directly
- **Rationale**: Support both single state and batch processing

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| All DQN parameters | See dqnAgentMethodology.md | Standard DQN hyperparameters |
| per_alpha | 0.6 | Prioritization exponent (0=uniform, 1=full priority) |
| per_beta_start | 0.4 | Initial importance sampling exponent |
| per_beta_end | 1.0 | Final importance sampling exponent |
| per_epsilon | 1e-6 | Small constant to ensure non-zero priority |

## Implementation Checklist

- [ ] All DQN implementation checklist items
- [ ] Implement dueling architecture (value + advantage streams)
- [ ] Implement advantage mean subtraction for identifiability
- [ ] Handle batch dimension (1D and 2D inputs)
- [ ] Implement double Q-learning (online selects, target evaluates)
- [ ] Implement PER buffer (SumTree for priority sampling)
- [ ] Calculate importance sampling weights
- [ ] Implement beta annealing schedule
- [ ] Update priorities based on TD-errors
- [ ] Track max_priority for new transitions
- [ ] Handle PER edge cases (empty buffer, zero priority, etc.)
- [ ] Apply importance sampling weights to loss
- [ ] Set `is_on_policy = False` and `needs_replay_learning = True`

