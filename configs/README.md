# Configuration Files

This directory contains YAML configuration files for experiments.

## Structure

Configuration files follow a hierarchical structure:

```yaml
experiment:
  name: experiment_name
  output_dir: results

agent:
  type: agent_type  # random, q_learning, dqn, rule_based_heuristic
  # Agent-specific hyperparameters

training:
  num_episodes: 1000
  max_steps_per_episode: 1000
  log_interval: 100
  save_interval: null

environment:
  reward_schema: sparse  # sparse, dense, decoupled-ila
  player_id: 0
  seed: 42
```

## Default Configuration

`default_config.yaml` provides a template configuration that can be used as a starting point for experiments.

## Agent-Specific Parameters

### Q-Learning Agent

```yaml
agent:
  type: q_learning
  learning_rate: 0.1
  discount_factor: 0.9
  epsilon: 0.1
  epsilon_decay: 0.9995
  min_epsilon: 0.01
```

### DQN Agent

```yaml
agent:
  type: dqn
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  batch_size: 32
  buffer_size: 80000
  target_update_freq: 1000
  per_alpha: 0.6
  per_beta_start: 0.4
  per_beta_end: 1.0
```

### Rule-Based Heuristic Agent

```yaml
agent:
  type: rule_based_heuristic
  debug_scores: false
```

## Usage

Specify a config file when running training:

```bash
python -m src.rl_agent_ludo.main --config configs/default_config.yaml
```

Override specific parameters via command line:

```bash
python -m src.rl_agent_ludo.main \
    --config configs/default_config.yaml \
    --experiment-name my_experiment \
    --num-episodes 2000
```

