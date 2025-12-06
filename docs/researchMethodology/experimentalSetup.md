# Experimental Setup and Research Methodology

## Overview

This document describes the complete experimental setup, training procedures, seed management, evaluation protocol, and blind run checklist for reproducing all experiments.

## Training Procedure

### Initialization Order

1. **Load Configuration**: Read YAML config file
2. **Set Seeds**: Initialize random seeds (Python, NumPy, PyTorch if available)
3. **Create Environment**: Initialize LudoEnv with seed and reward schema
4. **Create Agent**: Initialize agent with seed and hyperparameters
5. **Create Trainer**: Initialize trainer with env, agent, and config
6. **Run Training**: Execute training loop (on-policy or off-policy based on agent)

### Turn Order

- **Learning Agent**: Always `player_id=0`
- **Opponents**: Players 1, 2, 3 (random agents by default)
- **Turn Flow**:
  1. Agent's turn: Agent selects action, environment executes
  2. Opponent turns: Environment automatically handles (random moves)
  3. Repeat until game ends or max steps exceeded

### Episode Flow

```python
for episode in range(num_episodes):
    state = env.reset(seed=seed)
    done = False
    step = 0
    
    while not done and step < max_steps_per_episode:
        if env.current_player == agent.player_id:
            # Agent's turn
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            # Store experience, learn, log metrics
        else:
            # Opponent's turn (auto-handled)
            next_state, reward, terminated, truncated, info = env.step(0)
        
        state = next_state
        step += 1
        if terminated or truncated:
            done = True
    
    agent.on_episode_end()  # Epsilon decay, etc.
```

## Seed Management

### Seed Sources and Priority

Seeds are determined in priority order:
1. `config['training']['seed']` (highest priority)
2. `config['experiment']['seed']`
3. `env.seed`
4. `agent.seed`
5. `None` (non-deterministic)

### Seed Propagation

**What Gets Seeded**:
```python
random.seed(seed)           # Python random
np.random.seed(seed)        # NumPy random
torch.manual_seed(seed)     # PyTorch random (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # CUDA random
```

**When Seeds Are Set**:
- **Environment**: During `env.reset(seed=seed)` or `env.__init__(seed=seed)`
- **Agent**: During `agent.__init__(seed=seed)`
- **Trainer**: During `trainer._set_seeds(seed)` before training loop

### Recommended Seeds

For multi-run experiments, use different seeds:
- **5-run experiment**: [42, 123, 456, 789, 2024]
- **10-run experiment**: [42, 123, 456, 789, 2024, 1337, 2468, 3691, 4812, 5923]

### Seed in Output Directory

Output directory structure includes seed:
```
results/
  {agent_type}/
    reward_{reward_schema}/
      seed_{seed}/
        {experiment_name}_{timestamp}/
```

Example: `results/dqn/reward_dense/seed_42/dqn_dueling_augmented_raw_scaled_20251129_033043/`

## Episode Configuration

### DQN Agent

- **Total Episodes**: 75,000
- **Warmup Episodes**: 2,000 (no learning, just collecting experiences)
- **Min Buffer Size**: 1,000 (skip learning until buffer has 1,000 samples)
- **Max Steps per Episode**: 10,000 (safety limit)
- **Log Interval**: Every 2,000 episodes
- **Checkpoint Interval**: Every 5,000 episodes

### Q-Learning Agent

- **Total Episodes**: 30,000
- **Max Steps per Episode**: 10,000
- **Log Interval**: Every 1,000 episodes
- **Checkpoint Interval**: Every 2,000 episodes

### Random Agent (Baseline)

- **Total Episodes**: 1,000-10,000 (for validation)
- **No Learning**: No warmup or buffer requirements

## Agent-Specific Training Details

### DQN Training

- **Train Frequency**: Every N steps (default: 4, train every 4 steps)
- **Buffer Warmup**: Collect 1,000-2,000 experiences before learning
- **Target Network Updates**: Every 1,000 steps (copy online → target)
- **Learning Schedule**:
  - Episodes 0-2,000: Collect experiences only (warmup)
  - Episodes 2,000+: Start learning when buffer >= batch_size

### Q-Learning Training

- **Online Updates**: Immediate Q-table update per transition
- **No Buffer**: Updates happen during `push_to_replay_buffer()` call
- **Learning Schedule**: Start learning immediately (no warmup needed)

## Evaluation Protocol

### When to Evaluate

- **After Training**: Complete all training episodes first
- **Separate Evaluation Run**: Create new agent instance, load trained weights
- **Evaluation Mode**: Set `epsilon=0` (greedy, no exploration)

### Evaluation Episodes

- **Recommended**: 100-1,000 episodes
- **Metrics to Track**:
  - Win rate (fraction of games won)
  - Average episode reward
  - Average episode length
  - Standard deviation across episodes

### Statistical Reporting

Report results as:
- **Mean ± Standard Deviation** across multiple seeds
- **Example**: "Win Rate: 0.65 ± 0.08 (mean ± std across 5 seeds)"

## Hyperparameter Configurations

### DQN Default Configuration

```yaml
agent:
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  batch_size: 32
  buffer_size: 80000
  target_update_freq: 1000
  train_freq: 4
  state_representation: "orthogonal"  # or "augmented_raw"
  per_alpha: 0.6
  per_beta_start: 0.4
  per_beta_end: 1.0
  per_epsilon: 1e-6

training:
  num_episodes: 75000
  warmup_episodes: 2000
  min_buffer_size: 1000
```

### Q-Learning Default Configuration

```yaml
agent:
  learning_rate: 0.1
  discount_factor: 0.9
  epsilon: 0.1
  epsilon_decay: 0.9995
  min_epsilon: 0.01

training:
  num_episodes: 30000
```

### Reward Schema Options

- **"sparse"**: Only win/loss rewards (+100 win, -100 loss)
- **"dense"**: Enhanced dense rewards with progress-based bonuses
- **"context-aware"**: Context-aware reward scaling (for Q-Learning)

### State Representation Options

- **"orthogonal"**: 31-dimensional feature vector
- **"augmented_raw"**: 90-dimensional feature vector

## Output Directory Structure

```
results/
  {agent_type}/                    # dqn, q_learning, random, etc.
    reward_{reward_schema}/         # reward_sparse, reward_dense, etc.
      seed_{seed}/                  # seed_42, seed_123, etc.
        {experiment_name}_{timestamp}/
          {experiment_name}_episodes.csv
          {experiment_name}_score_debug.log (if enabled)
          partial_snapshots/ (if enabled)
```

## Edge Cases

### Seed Handling

#### None vs Integer
- **Scenario**: Seed parameter is `None`
- **Handling**: Use system random (non-deterministic)
- **Rationale**: Allows both deterministic (testing) and non-deterministic (baseline) operation

#### Seed Propagation
- **Scenario**: Multiple seed sources (training, experiment, env, agent)
- **Handling**: Use priority order (training > experiment > env > agent)
- **Rationale**: Training seed should override others for reproducibility

### Episode Counting

#### Resume from Checkpoint
- **Scenario**: Resuming training from saved checkpoint
- **Handling**: Extract episode number from filename: `agent_episode_{N}.pth` → N
- **Start Episode**: Set `start_episode = N`, continue from episode N+1
- **Rationale**: Maintain continuous episode numbering

### Turn Order

#### Agent Turn vs Opponent Turn
- **Scenario**: Multi-player game, agent is player 0
- **Handling**: 
  - Check `env.current_player == agent.player_id`
  - If agent's turn: Get action from agent, execute, learn
  - If opponent's turn: Environment handles automatically (random or specified opponent agent)
- **Rationale**: Agent only acts on its turn, opponents handled separately

### Terminal States

#### Game Won vs Max Steps Exceeded
- **Scenario**: Episode ends
- **Handling**:
  - `terminated = True`: Game won/lost (natural ending)
  - `truncated = True`: Max steps exceeded (timeout)
- **Rationale**: Distinguish between natural and forced episode endings

#### Terminal State Rewards
- **Scenario**: Episode ends (terminated or truncated)
- **Handling**: 
  - For DQN: `targets = rewards + (gamma * next_q * ~dones)` (no future reward if done)
  - For Q-Learning: `max_next_q = 0` if done (no future reward)
- **Rationale**: Terminal states have no future rewards

## Blind Run Checklist

### Before Starting Training

- [ ] **Set Seed**: Ensure seed is set in config or command line
- [ ] **Initialize Environment**: `env = LudoEnv(seed=seed, reward_schema=...)`
- [ ] **Initialize Agent**: `agent = Agent(seed=seed, ...)`
- [ ] **Set Training Seed**: `trainer._set_seeds(seed)` before training loop
- [ ] **Log Seed Value**: Seed should appear in output directory name
- [ ] **Save Config**: Copy config file to output directory for reference
- [ ] **Track Episode Count**: Initialize `episode_count = 0` (or from checkpoint if resuming)

### During Training

- [ ] **Respect Train Frequency**: Only train every N steps (if `train_freq > 1`)
- [ ] **Buffer Warmup**: Skip learning until `buffer.size >= min_buffer_size`
- [ ] **Target Network Updates**: Update every `target_update_freq` steps
- [ ] **Epsilon Decay**: Decay per episode (not per step)
- [ ] **Checkpoint Saving**: Save every `save_interval` episodes
- [ ] **Metrics Logging**: Log every `log_interval` episodes

### After Training

- [ ] **Save Final Model**: Save agent weights to checkpoint directory
- [ ] **Save Metrics**: Final metrics CSV and logs
- [ ] **Evaluation Run**: Run evaluation with `epsilon=0` (greedy)
- [ ] **Report Statistics**: Mean ± std across seeds

## Implementation Notes

- **Episode Numbering**: Start at 0, increment after each episode
- **Step Counting**: Increment after each agent action (not opponent actions)
- **Checkpoint Naming**: `agent_episode_{episode_number}.pth`
- **Resume Logic**: Extract episode number from checkpoint filename, start from that episode
- **Output Directory**: Include experiment name, timestamp, seed, reward schema for easy identification

