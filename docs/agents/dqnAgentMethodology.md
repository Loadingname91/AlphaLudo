# DQN Agent Methodology

## Theory

The Deep Q-Network (DQN) Agent uses a neural network to approximate Q-values, enabling learning in high-dimensional state spaces. It uses experience replay to break correlation between consecutive samples and a target network to stabilize training.

## Network Architecture

### Standard DQN Architecture

```
Input (state_dim) → Linear(hidden_dim) → ReLU → Linear(num_actions) → Q-values
```

**For Ludo**:
- Input dimension: 31 (orthogonal) or 90 (augmented_raw)
- Hidden dimension: 128 (orthogonal) or 256 (augmented_raw)
- Output: 4 Q-values (one per piece)

### Network Details

- **Input Layer**: `Linear(input_dim, hidden_dim)`
- **Hidden Layer**: `ReLU` activation
- **Output Layer**: `Linear(hidden_dim, num_actions)` → 4 Q-values

## State Representation

### Orthogonal State (31-dim)
- Per-piece features (20 dims): Progress, Safety, Home Corridor, Threat Distance, Kill Opportunity
- Global features (11 dims): Relative Progress, Pieces in Yard, Pieces Scored, Enemy Scored, Max Kill Potential, Dice One-Hot

### Augmented Raw State (90-dim)
- Global context (10 dims): Dice one-hot, Active player one-hot
- Token data (80 dims): 16 tokens × 5 features

## Action Selection

### Epsilon-Greedy Policy

```python
if random() < epsilon:
    action = random.choice(movable_pieces)  # Exploration
else:
    # Exploitation: select piece with highest Q-value
    q_values = network(state_vector)
    best_piece = argmax(q_values[movable_pieces])
    action = index_of(best_piece in movable_pieces)
```

**Important**: Network outputs Q-values per piece index (0-3), but environment expects action index into `movable_pieces`.

## Experience Replay

### Uniform Sampling

- Store transitions: `(state, action, reward, next_state, done)`
- Sample uniformly from buffer
- Break correlation between consecutive samples
- Enable learning from past experiences

### Buffer Management

- **Capacity**: 80,000 transitions (default)
- **Add**: New transitions added with max priority (for PER compatibility, but uniform sampling ignores priority)
- **Sample**: Uniform random sampling

## Learning: Bellman Update

### Standard DQN Update

```
Q(s,a) = Q(s,a) + α[r + γ × max Q_target(s',a') - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate (default: 0.0001)
- `γ` (gamma): Discount factor (default: 0.99)
- `Q_target`: Target network (frozen copy, updated periodically)

### Loss Function

**Huber Loss** (Smooth L1):
```python
loss = smooth_l1_loss(q_selected, targets)
```

Huber loss is less sensitive to outliers than MSE.

### Target Network

- **Purpose**: Stabilize training by using fixed targets
- **Update Frequency**: Every `target_update_freq` steps (default: 1000)
- **Update Method**: Copy weights from online network: `target_net.load_state_dict(online_net.state_dict())`

## Edge Cases

### Buffer Size < Batch Size
- **Scenario**: Insufficient samples in replay buffer
- **Handling**: Skip learning until `buffer.size >= batch_size`
- **Rationale**: Need enough samples for stable batch gradient
- **Warmup Period**: Typically 1000-2000 steps before learning starts

### Invalid Action Indices
- **Scenario**: Action index out of bounds or doesn't map to valid piece
- **Handling**: 
  - Map action index to piece index via `state.movable_pieces`
  - Fallback: If `movable_pieces` is None/empty, use all pieces (0-3)
- **Rationale**: Handle different state representations

### Empty Movable Pieces
- **Scenario**: `state.movable_pieces` is empty or None
- **Handling**: Fallback to `list(range(4))` (all pieces)
- **Rationale**: Should not happen, but provides graceful degradation

### Train Frequency
- **Scenario**: `train_freq > 1` (train every N steps, not every step)
- **Handling**: Check `if step_count % train_freq != 0: return`
- **Rationale**: Reduce computational load, train less frequently

### Target Network Update
- **Scenario**: Update target network at specific intervals
- **Handling**: `if step_count % target_update_freq == 0: update_target_net()`
- **Rationale**: Periodic updates stabilize learning (not every step)

### Device Handling
- **Scenario**: CPU vs GPU, tensor device mismatches
- **Handling**: 
  - Convert all tensors to same device: `.to(device)`
  - Ensure state vectors, actions, rewards, next_states all on same device
- **Rationale**: PyTorch requires all tensors in computation on same device

### Gradient Clipping
- **Scenario**: Exploding gradients
- **Handling**: `clip_grad_norm_(parameters, max_norm=10.0)`
- **Rationale**: Prevent gradient explosion, stabilize training

### State Representation Mismatch
- **Scenario**: Loading model with different `input_dim` than current agent
- **Handling**: 
  - Check `saved_input_dim != self.input_dim`
  - Print warning
  - Attempt load (will error if shapes don't match)
- **Rationale**: Handle model versioning, different state representations

### Network Initialization
- **Scenario**: First training step, target network sync
- **Handling**: 
  - Initialize target network: `target_net.load_state_dict(online_net.state_dict())`
  - Set target network to eval mode: `target_net.eval()`
- **Rationale**: Start with identical networks, target network doesn't train

### Terminal States
- **Scenario**: `done = True` (episode ended)
- **Handling**: `targets = rewards + (gamma * next_q * ~dones)`
- **Rationale**: No future reward if episode ended (multiply by `~dones`)

### Epsilon Decay
- **Schedule**: Per episode (not per step)
- **Formula**: `epsilon = max(epsilon_end, epsilon * epsilon_decay)`
- **Default**: Start 1.0, decay 0.995, end 0.01
- **Rationale**: Slow decay maintains exploration throughout training

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 0.0001 | Adam optimizer learning rate |
| gamma | 0.99 | Discount factor |
| epsilon_start | 1.0 | Initial exploration rate |
| epsilon_end | 0.01 | Final exploration rate |
| epsilon_decay | 0.995 | Epsilon decay per episode |
| batch_size | 32 | Batch size for training |
| buffer_size | 80,000 | Replay buffer capacity |
| target_update_freq | 1000 | Steps between target network updates |
| train_freq | 1 | Train every N steps |
| device | "cpu" | Device for computation |
| state_representation | "orthogonal" | "orthogonal" or "augmented_raw" |

## Implementation Checklist

- [ ] Initialize state abstractor (orthogonal or augmented_raw)
- [ ] Initialize neural network (input_dim, hidden_dim, num_actions=4)
- [ ] Initialize target network (copy of online network)
- [ ] Initialize optimizer (Adam)
- [ ] Initialize experience replay buffer (uniform sampling)
- [ ] Implement epsilon-greedy action selection
- [ ] Map action indices to piece indices correctly
- [ ] Convert states to feature vectors before adding to buffer
- [ ] Implement Bellman update with target network
- [ ] Handle buffer warmup (skip learning if `size < batch_size`)
- [ ] Respect train frequency (only train every N steps)
- [ ] Update target network periodically
- [ ] Handle device consistency (CPU/GPU)
- [ ] Clip gradients (max_norm=10.0)
- [ ] Implement epsilon decay per episode
- [ ] Implement save/load for model checkpoints
- [ ] Handle state representation mismatches in load
- [ ] Set `is_on_policy = False` and `needs_replay_learning = True`
- [ ] Implement optional score debugging

