# C4 Model - Level 3: Component Diagrams

## Overview

The **Component** diagrams zoom into individual containers, showing the components within them and how they interact. This document contains component diagrams for all major containers in the RL Agent Ludo system.

---

## 1. Training Application Container - Components

```mermaid
C4Component
    title Component Diagram: Training Application Container

    Container_Boundary(training_app, "Training Application") {
        Component(trainer, "Trainer", "Python Class", "Main orchestrator managing training loop, environment-agent interaction, and logging")
        Component(config_manager, "ConfigManager", "Python Class", "Loads and manages experiment configuration (agent type, hyperparameters, seeds)")
        Component(loop_on_policy, "OnPolicyLoop", "Python Method", "Training loop for on-policy algorithms (PPO)")
        Component(loop_off_policy, "OffPolicyLoop", "Python Method", "Training loop for off-policy algorithms (Q-Learning/DQN)")
        Component(seed_manager, "SeedManager", "Python Method", "Sets random seeds for reproducibility")
        Component(logger, "Logger", "Python Class", "Logging interface for console output")
    }

    System_Ext(agent_lib, "Agent Library")
    System_Ext(env_lib, "Environment Library")
    System_Ext(metrics, "Metrics Tracker")
    System_Ext(exp_logger, "Experiment Logger", "TensorBoard/WandB")

    Rel(trainer, config_manager, "Uses", "Python API")
    Rel(trainer, loop_on_policy, "Calls for PPO", "Python API")
    Rel(trainer, loop_off_policy, "Calls for Q-Learning/DQN", "Python API")
    Rel(trainer, seed_manager, "Calls", "Python API")
    Rel(trainer, logger, "Uses", "Python API")
    
    Rel(trainer, agent_lib, "Interacts with agents", "Agent Interface")
    Rel(trainer, env_lib, "Manages environment", "LudoEnv API")
    Rel(trainer, metrics, "Logs metrics", "MetricsTracker API")
    Rel(trainer, exp_logger, "Streams experiment logs", "HTTP/API")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Training Application Components

| Component | Responsibility | Key Methods/Attributes |
|-----------|---------------|----------------------|
| **Trainer** | Main orchestrator | `run()`, `env`, `agent`, `config`, `logger`, `experiment_logger` |
| **ConfigManager** | Configuration loading | `load_config()`, `get_agent_config()`, `get_training_config()` |
| **OnPolicyLoop** | On-policy training logic | `_run_on_policy_loop()`, handles PPO-style learning |
| **OffPolicyLoop** | Off-policy training logic | `_run_off_policy_loop()`, handles Q-Learning/DQN-style learning |
| **SeedManager** | Reproducibility | `_set_seeds()`, sets Python/Numpy/PyTorch seeds |
| **Logger** | Console logging | `info()`, `warning()`, `error()`, `debug()` |

---

## 2. Agent Library Container - Components

```mermaid
C4Component
    title Component Diagram: Agent Library Container

    Container_Boundary(agent_lib, "Agent Library") {
        Component(agent_interface, "Agent", "Abstract Base Class", "Interface defining methods all agents must implement")
        Component(agent_registry, "AgentRegistry", "Factory Pattern", "Constructs agent objects from configuration")
        
        Component(random_agent, "RandomAgent", "Concrete Agent", "Baseline agent selecting random valid actions")
        Component(tabular_q_agent, "TabularQAgent", "Concrete Agent", "Q-Learning with manual state abstraction")
        Component(td_agent, "TDAgent", "Concrete Agent", "TD(λ) with eligibility traces")
        Component(dqn_agent, "DQNAgent", "Concrete Agent", "Deep Q-Network with neural network approximation")
        Component(ppo_agent, "PPOAgent", "Concrete Agent", "Proximal Policy Optimization (on-policy)")
        Component(mcts_agent, "MCTSAgent", "Concrete Agent", "Monte-Carlo Tree Search with neural network")
        
        Component(replay_buffer, "ReplayBuffer", "Data Structure", "Stores experiences for off-policy learning (DQN)")
        Component(target_network, "TargetNetwork", "Neural Network", "Stable target Q-network for DQN")
        Component(policy_network, "PolicyNetwork", "Neural Network", "Policy network for PPO")
        Component(value_network, "ValueNetwork", "Neural Network", "Value network for PPO")
        Component(mcts_tree, "MCTSTree", "Data Structure", "Monte-Carlo search tree for MCTS agent")
    }

    System_Ext(ml_frameworks, "ML Frameworks", "PyTorch/TensorFlow")

    Rel(agent_registry, agent_interface, "Returns", "Agent Instance")
    Rel(agent_registry, random_agent, "Creates", "Python API")
    Rel(agent_registry, tabular_q_agent, "Creates", "Python API")
    Rel(agent_registry, td_agent, "Creates", "Python API")
    Rel(agent_registry, dqn_agent, "Creates", "Python API")
    Rel(agent_registry, ppo_agent, "Creates", "Python API")
    Rel(agent_registry, mcts_agent, "Creates", "Python API")
    
    Rel(agent_interface, random_agent, "Implemented by", "Inheritance")
    Rel(agent_interface, tabular_q_agent, "Implemented by", "Inheritance")
    Rel(agent_interface, td_agent, "Implemented by", "Inheritance")
    Rel(agent_interface, dqn_agent, "Implemented by", "Inheritance")
    Rel(agent_interface, ppo_agent, "Implemented by", "Inheritance")
    Rel(agent_interface, mcts_agent, "Implemented by", "Inheritance")
    
    Rel(dqn_agent, replay_buffer, "Uses", "Python API")
    Rel(dqn_agent, target_network, "Uses", "Python API")
    Rel(ppo_agent, policy_network, "Uses", "Python API")
    Rel(ppo_agent, value_network, "Uses", "Python API")
    Rel(mcts_agent, mcts_tree, "Uses", "Python API")
    
    Rel(dqn_agent, ml_frameworks, "Uses", "PyTorch API")
    Rel(ppo_agent, ml_frameworks, "Uses", "PyTorch API")
    Rel(mcts_agent, ml_frameworks, "Uses", "PyTorch API")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

### Agent Library Components

| Component | Responsibility | Key Methods/Attributes |
|-----------|---------------|----------------------|
| **Agent** | Abstract interface | `act()`, `learn_from_replay()`, `learn_from_rollout()`, `push_to_replay_buffer()`, `is_on_policy`, `needs_replay_learning` |
| **AgentRegistry** | Factory pattern | `create_agent(config)`, registers agent types |
| **RandomAgent** | Baseline | `act(state)` → random valid action |
| **TabularQAgent** | Q-Learning | `act()`, `learn_from_replay()`, Q-table, `alpha`, `gamma`, `epsilon` |
| **TDAgent** | TD(λ) | `act()`, `learn_from_replay()`, eligibility traces, `lambda` |
| **DQNAgent** | Deep Q-Network | `act()`, `learn_from_replay()`, `push_to_replay_buffer()`, replay buffer, target network |
| **PPOAgent** | Proximal Policy Optimization | `act()`, `learn_from_rollout()`, policy network, value network, `clip_range` |
| **MCTSAgent** | Monte-Carlo Tree Search | `act()`, `learn_from_rollout()`, MCTS tree, neural network evaluator |

---

## 3. Environment Library Container - Components

```mermaid
C4Component
    title Component Diagram: Environment Library Container

    Container_Boundary(env_lib, "Environment Library") {
        Component(ludo_env, "LudoEnv", "Gym-like Interface", "Hardware abstraction layer wrapping Ludopy")
        Component(state_dto, "State", "Data Transfer Object", "Immutable dataclass passing state from env to agent")
        Component(state_abstractor, "StateAbstractor", "State Abstraction Logic", "Converts raw game state to abstract representations")
        Component(opponent_manager, "OpponentManager", "Opponent Management", "Manages opponent agents and curriculum learning")
        Component(valid_actions, "ValidActionsManager", "Action Validation", "Filters and provides valid moves")
        Component(reward_shaper, "RewardShaper", "Strategy Pattern", "Implements reward shaping strategies")
        
        Component(sparse_reward, "SparseReward", "Reward Strategy", "Win/Loss only reward")
        Component(dense_reward, "DenseReward", "Reward Strategy", "Reward for all game events")
        Component(ila_reward, "ILAReward", "Reward Strategy", "Decoupled ILA reward components")
    }

    System_Ext(ludopy, "Ludopy Library")
    System_Ext(agent_lib, "Agent Library")

    Rel(ludo_env, state_dto, "Creates", "Python API")
    Rel(ludo_env, state_abstractor, "Uses", "Python API")
    Rel(ludo_env, opponent_manager, "Uses", "Python API")
    Rel(ludo_env, valid_actions, "Uses", "Python API")
    Rel(ludo_env, reward_shaper, "Uses", "Python API")
    
    Rel(ludo_env, ludopy, "Wraps", "Python API")
    Rel(opponent_manager, agent_lib, "Creates opponents", "Agent Interface")
    
    Rel(reward_shaper, sparse_reward, "Strategy", "Strategy Pattern")
    Rel(reward_shaper, dense_reward, "Strategy", "Strategy Pattern")
    Rel(reward_shaper, ila_reward, "Strategy", "Strategy Pattern")
    
    Rel(state_abstractor, state_dto, "Populates", "Python API")
    Rel(valid_actions, state_dto, "Populates", "Python API")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

### Environment Library Components

| Component | Responsibility | Key Methods/Attributes |
|-----------|---------------|----------------------|
| **LudoEnv** | HAL abstraction | `reset()`, `step()`, `get_valid_actions()`, `game`, `opponent_agents`, `opponent_schedule`, `player_id_map` |
| **State** | Immutable DTO | `full_vector` (NumPy array), `abstract_state` (hashable tuple), `valid_moves` (list), `dice_roll` (int) |
| **StateAbstractor** | State conversion | `_get_full_state_vector()`, `_get_abstract_state()`, `_get_observation()` |
| **OpponentManager** | Opponent handling | Manages `opponent_agents` list, `opponent_schedule` for curriculum learning |
| **ValidActionsManager** | Action filtering | `get_valid_actions()`, filters moves based on game rules |
| **RewardShaper** | Reward strategy | `get_reward(game_events)` → (reward, ila_components), `schema` |
| **SparseReward** | Sparse strategy | Reward only on win/loss |
| **DenseReward** | Dense strategy | Reward for piece moves, captures, home entry, etc. |
| **ILAReward** | ILA strategy | Decoupled individual learning algorithm components |

---

## 4. Metrics & Analysis Container - Components

```mermaid
C4Component
    title Component Diagram: Metrics & Analysis Containers

    Container_Boundary(metrics_tracker, "Metrics Tracker Container") {
        Component(metrics_collector, "MetricsTracker", "Python Class", "Collects raw training metrics during training")
        Component(episode_recorder, "EpisodeRecorder", "Data Structure", "Records episode-level metrics")
        Component(step_recorder, "StepRecorder", "Data Structure", "Records step-level metrics")
        Component(metric_exporter, "MetricExporter", "File Writer", "Exports metrics to JSON/CSV")
    }
    
    Container_Boundary(analysis_service, "Analysis Service Container") {
        Component(analysis_runner, "AnalysisRunner", "Python Script", "Main entry point for analysis")
        Component(data_loader, "DataLoader", "Pandas Reader", "Loads raw metrics from JSON/CSV")
        Component(policy_analyzer, "PolicyAnalyzer", "Analysis Module", "Policy & behavioral analysis")
        Component(stability_analyzer, "StabilityAnalyzer", "Analysis Module", "Stability & convergence analysis")
        Component(robustness_analyzer, "RobustnessAnalyzer", "Analysis Module", "Robustness & generalization analysis")
        Component(computational_analyzer, "ComputationalAnalyzer", "Analysis Module", "Computational & scalability analysis")
        Component(hparam_analyzer, "HyperparameterAnalyzer", "Analysis Module", "Hyperparameter sensitivity analysis")
        Component(plot_generator, "PlotGenerator", "Matplotlib/Seaborn", "Generates 5-point analysis plots")
        Component(report_generator, "ReportGenerator", "Report Builder", "Synthesizes final comparative report")
    }

    System_Ext(metrics_storage, "Metrics Storage", "JSON/CSV Files")
    System_Ext(report_output, "Report Output", "PDF/HTML")

    Rel(metrics_collector, episode_recorder, "Records to", "Python API")
    Rel(metrics_collector, step_recorder, "Records to", "Python API")
    Rel(metric_exporter, episode_recorder, "Reads from", "Python API")
    Rel(metric_exporter, step_recorder, "Reads from", "Python API")
    Rel(metric_exporter, metrics_storage, "Writes to", "File I/O")
    
    Rel(analysis_runner, data_loader, "Uses", "Python API")
    Rel(data_loader, metrics_storage, "Reads from", "File I/O")
    
    Rel(analysis_runner, policy_analyzer, "Calls", "Python API")
    Rel(analysis_runner, stability_analyzer, "Calls", "Python API")
    Rel(analysis_runner, robustness_analyzer, "Calls", "Python API")
    Rel(analysis_runner, computational_analyzer, "Calls", "Python API")
    Rel(analysis_runner, hparam_analyzer, "Calls", "Python API")
    
    Rel(policy_analyzer, plot_generator, "Uses", "Python API")
    Rel(stability_analyzer, plot_generator, "Uses", "Python API")
    Rel(robustness_analyzer, plot_generator, "Uses", "Python API")
    Rel(computational_analyzer, plot_generator, "Uses", "Python API")
    Rel(hparam_analyzer, plot_generator, "Uses", "Python API")
    
    Rel(analysis_runner, report_generator, "Uses", "Python API")
    Rel(report_generator, plot_generator, "Includes plots from", "Python API")
    Rel(report_generator, report_output, "Generates", "File I/O")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

### Metrics & Analysis Components

#### Metrics Tracker Components

| Component | Responsibility | Key Methods/Attributes |
|-----------|---------------|----------------------|
| **MetricsTracker** | Main collector | `log_metrics()`, `save_metrics()`, lightweight (no pandas) |
| **EpisodeRecorder** | Episode data | Lists/dicts storing episode-level metrics |
| **StepRecorder** | Step data | Lists/dicts storing step-level metrics |
| **MetricExporter** | File export | `export_to_json()`, `export_to_csv()` |

#### Analysis Service Components

| Component | Responsibility | Key Methods/Attributes |
|-----------|---------------|----------------------|
| **AnalysisRunner** | Main script | `run_analysis()`, orchestrates 5-point analysis |
| **DataLoader** | Data loading | `load_metrics()`, pandas DataFrame creation |
| **PolicyAnalyzer** | Policy analysis | Aggression, defense, efficiency metrics |
| **StabilityAnalyzer** | Stability analysis | Q-value variance, win-rate CI, convergence curves |
| **RobustnessAnalyzer** | Robustness analysis | Opponent swap test, IQL flaw detection, generalization |
| **ComputationalAnalyzer** | Computational analysis | Sample efficiency, inference time, memory usage |
| **HyperparameterAnalyzer** | Hparam sensitivity | Win rate vs. hyperparameter plots |
| **PlotGenerator** | Visualization | Matplotlib/Seaborn plot generation |
| **ReportGenerator** | Report synthesis | Final comparative report generation |

---

## Component Interaction Summary

### Training Flow
```
Trainer → LudoEnv → State → Agent
    ↓
RewardShaper → Reward
    ↓
MetricsTracker → Metrics Storage
```

### Learning Flow
```
Agent.act(State) → Action
LudoEnv.step(Action) → (State, Reward)
    ↓
Agent.learn_from_replay() OR Agent.learn_from_rollout()
```

### Analysis Flow
```
DataLoader → Raw Metrics
    ↓
5 Analyzers → Insights
    ↓
PlotGenerator → Visualizations
    ↓
ReportGenerator → Final Report
```

---

## Next Level

See [C4 Level 4: Code Diagrams](./c4-level4-code.md) for code-level interactions and sequence diagrams for specific scenarios.

