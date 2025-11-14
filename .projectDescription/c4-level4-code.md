# C4 Model - Level 4: Code Diagrams

## Overview

The **Code** diagrams show the internal structure of components, including classes, methods, functions, and their interactions. This level is the most granular and shows how code actually executes for specific scenarios.

---

## Scenario 1: On-Policy Training Loop (PPO Agent)

### Sequence Diagram: On-Policy Training Episode

```mermaid
sequenceDiagram
    participant Main as Main Entry Point
    participant Trainer as Trainer
    participant ConfigMgr as ConfigManager
    participant SeedMgr as SeedManager
    participant Logger as Logger
    participant LudoEnv as LudoEnv
    participant State as State DTO
    participant RewardShaper as RewardShaper
    participant PPOAgent as PPOAgent
    participant PolicyNet as PolicyNetwork
    participant ValueNet as ValueNetwork
    participant Metrics as MetricsTracker
    participant ExpLogger as ExperimentLogger (TensorBoard)

    Main->>Trainer: run(config_path)
    Trainer->>ConfigMgr: load_config(config_path)
    ConfigMgr-->>Trainer: config dict
    
    Trainer->>SeedMgr: _set_seeds(seed)
    SeedMgr->>SeedMgr: random.seed(seed)
    SeedMgr->>SeedMgr: np.random.seed(seed)
    SeedMgr->>SeedMgr: torch.manual_seed(seed)
    
    Trainer->>AgentRegistry: create_agent(config['agent'])
    AgentRegistry->>PPOAgent: __init__(config)
    PPOAgent->>PolicyNet: __init__(state_dim, action_dim)
    PPOAgent->>ValueNet: __init__(state_dim)
    PPOAgent-->>Trainer: agent instance
    
    Trainer->>LudoEnv: __init__(config['env'])
    LudoEnv->>OpponentManager: __init__(opponent_agents, schedule)
    LudoEnv-->>Trainer: env instance
    
    Trainer->>Metrics: __init__(experiment_name)
    Trainer->>ExpLogger: init(experiment_name)
    
    loop Training Episode
        Trainer->>LudoEnv: reset()
        LudoEnv->>Ludopy: __init__()
        LudoEnv->>StateAbstractor: _get_full_state_vector()
        LudoEnv->>StateAbstractor: _get_abstract_state()
        LudoEnv->>ValidActionsManager: get_valid_actions()
        LudoEnv->>State: State(full_vector, abstract_state, valid_moves, dice_roll)
        LudoEnv-->>Trainer: initial_state (State object)
        
        Note over Trainer,PPOAgent: Rollout Collection Phase
        loop Rollout Steps (N steps)
            Trainer->>PPOAgent: act(state)
            PPOAgent->>PolicyNet: forward(state.full_vector)
            PolicyNet-->>PPOAgent: action_probs
            PPOAgent->>PPOAgent: sample_action(action_probs, state.valid_moves)
            PPOAgent-->>Trainer: action
            
            Trainer->>LudoEnv: step(action)
            LudoEnv->>Ludopy: move_piece(player_id, piece_id)
            LudoEnv->>Ludopy: get_game_state()
            LudoEnv->>RewardShaper: get_reward(game_events)
            RewardShaper->>RewardShaper: _compute_reward(game_events)
            RewardShaper-->>LudoEnv: (reward, ila_components)
            LudoEnv->>StateAbstractor: _get_full_state_vector()
            LudoEnv->>StateAbstractor: _get_abstract_state()
            LudoEnv->>ValidActionsManager: get_valid_actions()
            LudoEnv->>State: State(full_vector, abstract_state, valid_moves, dice_roll)
            LudoEnv-->>Trainer: (next_state, reward, done, info)
            
            Trainer->>PPOAgent: push_to_replay_buffer(state, action, reward, next_state, done)
            PPOAgent->>PPOAgent: rollout_buffer.append(experience)
            
            Trainer->>Metrics: log_metrics(state, action, reward, info)
            Trainer->>ExpLogger: log_scalar("reward", reward, step)
            
            alt Episode Done
                Trainer->>LudoEnv: reset()
            end
        end
        
        Note over Trainer,PPOAgent: Learning Phase
        Trainer->>PPOAgent: learn_from_rollout(rollout_buffer)
        PPOAgent->>PPOAgent: compute_advantages(rollout_buffer)
        PPOAgent->>PPOAgent: normalize_advantages()
        
        loop PPO Update Steps (K epochs)
            PPOAgent->>PolicyNet: forward(states)
            PolicyNet-->>PPOAgent: new_action_probs
            PPOAgent->>PPOAgent: compute_policy_loss(new_probs, old_probs, advantages)
            PPOAgent->>PPOAgent: clip_policy_loss(loss, clip_range)
            PPOAgent->>ValueNet: forward(states)
            ValueNet-->>PPOAgent: values
            PPOAgent->>PPOAgent: compute_value_loss(values, targets)
            PPOAgent->>PPOAgent: compute_entropy_loss(new_probs)
            PPOAgent->>PPOAgent: total_loss = policy_loss + value_loss - entropy_loss
            PPOAgent->>PolicyNet: backward(total_loss)
            PPOAgent->>ValueNet: backward(total_loss)
        end
    end
    
    Trainer->>Metrics: save_metrics()
    Metrics->>MetricExporter: export_to_json(metrics, "experiment_metrics.json")
    Metrics->>MetricExporter: export_to_csv(episodes_df, "episodes.csv")
```

### Code-Level Class Structure: PPOAgent

```python
class PPOAgent(Agent):
    def __init__(self, config):
        self.is_on_policy = True
        self.needs_replay_learning = False
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        self.rollout_buffer = []
        self.clip_range = config['clip_range']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
    
    def act(self, state: State) -> int:
        state_vector = state.full_vector
        action_probs = self.policy_network.forward(state_vector)
        return self._sample_action(action_probs, state.valid_moves)
    
    def learn_from_rollout(self, rollout_buffer: List[Experience]):
        advantages = self._compute_advantages(rollout_buffer)
        normalized_advantages = self._normalize(advantages)
        
        for epoch in range(self.n_epochs):
            policy_loss = self._compute_policy_loss(rollout_buffer, normalized_advantages)
            value_loss = self._compute_value_loss(rollout_buffer)
            entropy_loss = self._compute_entropy_loss(rollout_buffer)
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            self._update_networks(total_loss)
```

---

## Scenario 2: Off-Policy Training Loop (DQN Agent)

### Sequence Diagram: Off-Policy Training with Experience Replay

```mermaid
sequenceDiagram
    participant Trainer as Trainer
    participant LudoEnv as LudoEnv
    participant State as State DTO
    participant DQNAgent as DQNAgent
    participant QNetwork as QNetwork
    participant TargetNetwork as TargetNetwork
    participant ReplayBuffer as ReplayBuffer
    participant Metrics as MetricsTracker

    Trainer->>LudoEnv: reset()
    LudoEnv-->>Trainer: initial_state
    
    loop Training Episode
        Trainer->>DQNAgent: act(state, epsilon)
        DQNAgent->>DQNAgent: _epsilon_greedy(epsilon)
        
        alt Exploration (random)
            DQNAgent->>DQNAgent: random.choice(state.valid_moves)
        else Exploitation
            DQNAgent->>QNetwork: forward(state.full_vector)
            QNetwork-->>DQNAgent: q_values
            DQNAgent->>DQNAgent: mask_invalid_actions(q_values, state.valid_moves)
            DQNAgent->>DQNAgent: argmax(q_values)
        end
        DQNAgent-->>Trainer: action
        
        Trainer->>LudoEnv: step(action)
        LudoEnv-->>Trainer: (next_state, reward, done, info)
        
        Trainer->>DQNAgent: push_to_replay_buffer(state, action, reward, next_state, done)
        DQNAgent->>ReplayBuffer: add(state, action, reward, next_state, done)
        
        alt Replay Buffer Ready (size > batch_size)
            Trainer->>DQNAgent: learn_from_replay()
            DQNAgent->>ReplayBuffer: sample(batch_size)
            ReplayBuffer-->>DQNAgent: batch (states, actions, rewards, next_states, dones)
            
            DQNAgent->>QNetwork: forward(states)
            QNetwork-->>DQNAgent: q_values_current
            
            DQNAgent->>QNetwork: forward(next_states)
            QNetwork-->>DQNAgent: q_values_next
            
            DQNAgent->>TargetNetwork: forward(next_states)
            TargetNetwork-->>DQNAgent: q_values_target
            
            DQNAgent->>DQNAgent: compute_targets(rewards, q_values_target, dones, gamma)
            DQNAgent->>DQNAgent: q_targets = rewards + gamma * q_values_target * (1 - dones)
            
            DQNAgent->>DQNAgent: q_expected = q_values_current.gather(1, actions)
            DQNAgent->>DQNAgent: loss = MSE(q_expected, q_targets)
            
            DQNAgent->>QNetwork: backward(loss)
            DQNAgent->>QNetwork: optimizer.step()
            
            alt Update Target Network (every N steps)
                DQNAgent->>TargetNetwork: load_state_dict(QNetwork.state_dict())
            end
        end
        
        Trainer->>Metrics: log_metrics(state, action, reward, info)
    end
```

### Code-Level Class Structure: DQNAgent

```python
class DQNAgent(Agent):
    def __init__(self, config):
        self.is_on_policy = False
        self.needs_replay_learning = True
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(capacity=config['buffer_size'])
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update_freq = config['target_update_freq']
    
    def act(self, state: State, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.choice(state.valid_moves)
        else:
            q_values = self.q_network.forward(state.full_vector)
            masked_q = self._mask_invalid_actions(q_values, state.valid_moves)
            return masked_q.argmax().item()
    
    def push_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def learn_from_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        q_current = self.q_network.forward(states).gather(1, actions)
        q_next = self.target_network.forward(next_states).max(1)[0].detach()
        q_targets = rewards + self.gamma * q_next * (1 - dones)
        
        loss = F.mse_loss(q_current, q_targets.unsqueeze(1))
        self._update_network(loss)
        
        if self.step_count % self.target_update_freq == 0:
            self._update_target_network()
```

---

## Scenario 3: State Abstraction Process

### Sequence Diagram: State Creation and Abstraction

```mermaid
sequenceDiagram
    participant LudoEnv as LudoEnv
    participant Ludopy as Ludopy Library
    participant StateAbstractor as StateAbstractor
    participant ValidActions as ValidActionsManager
    participant State as State DTO

    LudoEnv->>Ludopy: get_game_state()
    Ludopy-->>LudoEnv: raw_game_state (dict)
    
    Note over LudoEnv,State: Full Vector Creation (for Neural Networks)
    LudoEnv->>StateAbstractor: _get_full_state_vector(raw_game_state)
    StateAbstractor->>StateAbstractor: extract_piece_positions(raw_game_state)
    StateAbstractor->>StateAbstractor: extract_dice_roll(raw_game_state)
    StateAbstractor->>StateAbstractor: extract_player_positions(raw_game_state)
    StateAbstractor->>StateAbstractor: extract_safe_zones(raw_game_state)
    StateAbstractor->>StateAbstractor: encode_to_vector(positions, dice, safe_zones)
    StateAbstractor-->>LudoEnv: full_vector (numpy.ndarray, shape=[state_dim])
    
    Note over LudoEnv,State: Abstract State Creation (for Tabular Methods)
    LudoEnv->>StateAbstractor: _get_abstract_state(raw_game_state)
    StateAbstractor->>StateAbstractor: compute_piece_distances(raw_game_state)
    StateAbstractor->>StateAbstractor: compute_relative_positions(raw_game_state)
    StateAbstractor->>StateAbstractor: discretize_positions(positions)
    StateAbstractor->>StateAbstractor: create_hashable_tuple(discretized_features)
    StateAbstractor-->>LudoEnv: abstract_state (tuple, hashable)
    
    Note over LudoEnv,State: Valid Actions Extraction
    LudoEnv->>ValidActions: get_valid_actions(raw_game_state, dice_roll)
    ValidActions->>ValidActions: check_piece_in_start(piece)
    ValidActions->>ValidActions: check_piece_in_home(piece)
    ValidActions->>ValidActions: check_move_validity(piece, target_position)
    ValidActions->>ValidActions: filter_valid_moves(pieces, dice_roll)
    ValidActions-->>LudoEnv: valid_moves (list[int])
    
    LudoEnv->>Ludopy: get_dice_roll()
    Ludopy-->>LudoEnv: dice_roll (int)
    
    LudoEnv->>State: State(full_vector=full_vector,<br/>abstract_state=abstract_state,<br/>valid_moves=valid_moves,<br/>dice_roll=dice_roll)
    State-->>LudoEnv: state_instance
```

### Code-Level Class Structure: StateAbstractor

```python
@dataclass(frozen=True)
class State:
    full_vector: np.ndarray  # For neural networks
    abstract_state: tuple    # For tabular methods (hashable)
    valid_moves: List[int]   # List of valid action indices
    dice_roll: int           # Current dice roll

class StateAbstractor:
    def __init__(self, state_dim: int, discretization_levels: int):
        self.state_dim = state_dim
        self.discretization_levels = discretization_levels
    
    def _get_full_state_vector(self, raw_game_state: dict) -> np.ndarray:
        """Creates continuous state vector for neural networks."""
        features = []
        
        # Extract piece positions (4 players × 4 pieces = 16 positions)
        for player_id in range(4):
            for piece_id in range(4):
                position = raw_game_state['pieces'][player_id][piece_id]
                features.extend([position, self._is_in_safe_zone(position)])
        
        # Extract dice roll
        features.append(raw_game_state['dice_roll'])
        
        # Extract game phase indicators
        features.append(raw_game_state['current_player'])
        features.extend(self._get_player_progress(raw_game_state))
        
        return np.array(features, dtype=np.float32)
    
    def _get_abstract_state(self, raw_game_state: dict) -> tuple:
        """Creates discrete, hashable state for tabular methods."""
        # Discretize piece positions
        discretized = []
        for player_id in range(4):
            player_pieces = []
            for piece_id in range(4):
                position = raw_game_state['pieces'][player_id][piece_id]
                # Discretize to bins: [0-15, 16-31, 32-47, 48-63, home, start]
                if position == -1:  # Start
                    discrete_pos = -1
                elif position == 999:  # Home
                    discrete_pos = 999
                else:
                    discrete_pos = position // self.discretization_levels
                player_pieces.append(discrete_pos)
            discretized.append(tuple(sorted(player_pieces)))
        
        # Add dice roll and current player
        discretized.append(raw_game_state['dice_roll'])
        discretized.append(raw_game_state['current_player'])
        
        return tuple(discretized)
```

---

## Scenario 4: Reward Shaping Process

### Sequence Diagram: Reward Calculation with Strategy Pattern

```mermaid
sequenceDiagram
    participant LudoEnv as LudoEnv
    participant Ludopy as Ludopy Library
    participant RewardShaper as RewardShaper
    participant SparseReward as SparseReward Strategy
    participant DenseReward as DenseReward Strategy
    participant ILAReward as ILAReward Strategy

    LudoEnv->>Ludopy: execute_action(player_id, piece_id)
    Ludopy->>Ludopy: move_piece()
    Ludopy-->>LudoEnv: game_events (dict)
    
    Note over LudoEnv,ILAReward: game_events contains:<br/>- piece_moved<br/>- piece_captured<br/>- piece_entered_home<br/>- player_won<br/>- etc.
    
    LudoEnv->>RewardShaper: get_reward(game_events)
    
    alt Reward Schema: "sparse"
        RewardShaper->>SparseReward: get_reward(game_events)
        SparseReward->>SparseReward: check_win(game_events)
        alt Player Won
            SparseReward-->>RewardShaper: reward = 100, ila_components = {}
        else Game Lost
            SparseReward-->>RewardShaper: reward = -100, ila_components = {}
        else Ongoing
            SparseReward-->>RewardShaper: reward = 0, ila_components = {}
        end
    
    else Reward Schema: "dense"
        RewardShaper->>DenseReward: get_reward(game_events)
        DenseReward->>DenseReward: reward = 0
        alt Piece Moved Forward
            DenseReward->>DenseReward: reward += 1.0
        end
        alt Piece Captured Opponent
            DenseReward->>DenseReward: reward += 10.0
        end
        alt Piece Entered Home
            DenseReward->>DenseReward: reward += 20.0
        end
        alt Player Won
            DenseReward->>DenseReward: reward += 100.0
        end
        alt Opponent Won
            DenseReward->>DenseReward: reward -= 100.0
        end
        DenseReward-->>RewardShaper: reward, ila_components = {}
    
    else Reward Schema: "decoupled-ila"
        RewardShaper->>ILAReward: get_reward(game_events)
        ILAReward->>ILAReward: ila_components = {}
        ILAReward->>ILAReward: ila_components['piece_movement'] = compute_piece_movement_reward()
        ILAReward->>ILAReward: ila_components['capture'] = compute_capture_reward()
        ILAReward->>ILAReward: ila_components['home_entry'] = compute_home_entry_reward()
        ILAReward->>ILAReward: ila_components['win'] = compute_win_reward()
        ILAReward->>ILAReward: total_reward = sum(ila_components.values())
        ILAReward-->>RewardShaper: reward, ila_components
    
    end
    
    RewardShaper-->>LudoEnv: (reward, ila_components)
```

### Code-Level Class Structure: RewardShaper

```python
class RewardShaper(ABC):
    @abstractmethod
    def get_reward(self, game_events: dict) -> tuple[float, dict]:
        """Returns (net_reward, ila_components_dict)."""
        pass

class SparseReward(RewardShaper):
    def get_reward(self, game_events: dict) -> tuple[float, dict]:
        if game_events.get('player_won'):
            return 100.0, {}
        elif game_events.get('opponent_won'):
            return -100.0, {}
        else:
            return 0.0, {}

class DenseReward(RewardShaper):
    def get_reward(self, game_events: dict) -> tuple[float, dict]:
        reward = 0.0
        
        if game_events.get('piece_moved'):
            reward += 1.0
        
        if game_events.get('piece_captured'):
            reward += 10.0
        
        if game_events.get('piece_entered_home'):
            reward += 20.0
        
        if game_events.get('player_won'):
            reward += 100.0
        elif game_events.get('opponent_won'):
            reward -= 100.0
        
        return reward, {}

class ILAReward(RewardShaper):
    def get_reward(self, game_events: dict) -> tuple[float, dict]:
        ila_components = {
            'piece_movement': self._compute_movement_reward(game_events),
            'capture': self._compute_capture_reward(game_events),
            'home_entry': self._compute_home_entry_reward(game_events),
            'win': self._compute_win_reward(game_events),
        }
        
        total_reward = sum(ila_components.values())
        return total_reward, ila_components

class RewardShaperFactory:
    @staticmethod
    def create(schema: str) -> RewardShaper:
        strategies = {
            'sparse': SparseReward,
            'dense': DenseReward,
            'decoupled-ila': ILAReward,
        }
        return strategies[schema]()
```

---

## Scenario 5: Metrics Collection Process

### Sequence Diagram: Metrics Logging and Export

```mermaid
sequenceDiagram
    participant Trainer as Trainer
    participant Metrics as MetricsTracker
    participant EpisodeRecorder as EpisodeRecorder
    participant StepRecorder as StepRecorder
    participant Exporter as MetricExporter
    participant FileSystem as File System

    Trainer->>Metrics: log_metrics(state, action, reward, info)
    
    Note over Metrics,StepRecorder: Step-Level Metrics
    Metrics->>StepRecorder: record_step(step_data)
    StepRecorder->>StepRecorder: self.steps.append({<br/>  'episode': episode_num,<br/>  'step': step_num,<br/>  'state': state.abstract_state,<br/>  'action': action,<br/>  'reward': reward,<br/>  'q_value': info.get('q_value'),<br/>  'epsilon': info.get('epsilon')<br/>})
    
    alt Episode Done
        Note over Metrics,EpisodeRecorder: Episode-Level Metrics
        Trainer->>Metrics: log_episode(episode_data)
        Metrics->>EpisodeRecorder: record_episode(episode_data)
        EpisodeRecorder->>EpisodeRecorder: self.episodes.append({<br/>  'episode': episode_num,<br/>  'total_reward': sum(episode_rewards),<br/>  'steps': step_count,<br/>  'won': info.get('won', False),<br/>  'win_rate': cumulative_win_rate,<br/>  'avg_q_value': mean_q_value<br/>})
    end
    
    Note over Metrics,FileSystem: Export Phase (after training)
    Trainer->>Metrics: save_metrics()
    Metrics->>Exporter: export_to_json(episodes, "episodes.json")
    Exporter->>FileSystem: write JSON file
    
    Metrics->>Exporter: export_to_csv(episodes, "episodes.csv")
    Exporter->>Exporter: convert_to_dataframe(episodes)
    Exporter->>FileSystem: write CSV file
    
    Metrics->>Exporter: export_to_json(steps, "steps.json")
    Exporter->>FileSystem: write JSON file
```

### Code-Level Class Structure: MetricsTracker

```python
class MetricsTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.episode_recorder = EpisodeRecorder()
        self.step_recorder = StepRecorder()
        self.exporter = MetricExporter()
        self.current_episode = 0
        self.current_step = 0
    
    def log_metrics(self, state: State, action: int, reward: float, info: dict):
        step_data = {
            'episode': self.current_episode,
            'step': self.current_step,
            'state': state.abstract_state,
            'action': action,
            'reward': reward,
            **info  # q_value, epsilon, etc.
        }
        self.step_recorder.record_step(step_data)
        self.current_step += 1
    
    def log_episode(self, episode_data: dict):
        episode_data['episode'] = self.current_episode
        self.episode_recorder.record_episode(episode_data)
        self.current_episode += 1
        self.current_step = 0
    
    def save_metrics(self, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Export episodes
        self.exporter.export_to_json(
            self.episode_recorder.episodes,
            f"{output_dir}/{self.experiment_name}_episodes.json"
        )
        self.exporter.export_to_csv(
            self.episode_recorder.episodes,
            f"{output_dir}/{self.experiment_name}_episodes.csv"
        )
        
        # Export steps (optional, can be large)
        if self.step_recorder.steps:
            self.exporter.export_to_json(
                self.step_recorder.steps,
                f"{output_dir}/{self.experiment_name}_steps.json"
            )

class EpisodeRecorder:
    def __init__(self):
        self.episodes = []
    
    def record_episode(self, episode_data: dict):
        self.episodes.append(episode_data)

class StepRecorder:
    def __init__(self):
        self.steps = []
    
    def record_step(self, step_data: dict):
        self.steps.append(step_data)

class MetricExporter:
    @staticmethod
    def export_to_json(data: list, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def export_to_csv(data: list, filepath: str):
        # Convert dict list to CSV (minimal pandas usage in actual implementation)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
```

---

## Scenario 6: Analysis Execution Process

### Sequence Diagram: 5-Point Analysis Framework

```mermaid
sequenceDiagram
    participant DataScientist as Data Scientist
    participant AnalysisRunner as AnalysisRunner
    participant DataLoader as DataLoader
    participant PolicyAnalyzer as PolicyAnalyzer
    participant StabilityAnalyzer as StabilityAnalyzer
    participant RobustnessAnalyzer as RobustnessAnalyzer
    participant ComputationalAnalyzer as ComputationalAnalyzer
    participant HyperparameterAnalyzer as HyperparameterAnalyzer
    participant PlotGenerator as PlotGenerator
    participant ReportGenerator as ReportGenerator
    participant FileSystem as File System

    DataScientist->>AnalysisRunner: run_analysis(config_path)
    AnalysisRunner->>DataLoader: load_metrics(config['metrics_dir'])
    DataLoader->>FileSystem: read JSON/CSV files
    FileSystem-->>DataLoader: raw_metrics_data
    DataLoader->>DataLoader: convert_to_dataframe(episodes, steps)
    DataLoader-->>AnalysisRunner: metrics_df (pandas.DataFrame)
    
    Note over AnalysisRunner,HyperparameterAnalyzer: 5-Point Analysis Execution
    
    AnalysisRunner->>PolicyAnalyzer: analyze(metrics_df)
    PolicyAnalyzer->>PolicyAnalyzer: compute_aggression_metrics()
    PolicyAnalyzer->>PolicyAnalyzer: compute_defense_metrics()
    PolicyAnalyzer->>PolicyAnalyzer: compute_efficiency_metrics()
    PolicyAnalyzer->>PlotGenerator: plot_policy_metrics(metrics)
    PolicyAnalyzer-->>AnalysisRunner: policy_insights, policy_plots
    
    AnalysisRunner->>StabilityAnalyzer: analyze(metrics_df)
    StabilityAnalyzer->>StabilityAnalyzer: compute_q_value_variance()
    StabilityAnalyzer->>StabilityAnalyzer: compute_win_rate_ci()
    StabilityAnalyzer->>StabilityAnalyzer: compute_convergence_curves()
    StabilityAnalyzer->>PlotGenerator: plot_stability_metrics(metrics)
    StabilityAnalyzer-->>AnalysisRunner: stability_insights, stability_plots
    
    AnalysisRunner->>RobustnessAnalyzer: analyze(metrics_df)
    RobustnessAnalyzer->>RobustnessAnalyzer: run_opponent_swap_test()
    RobustnessAnalyzer->>RobustnessAnalyzer: detect_iql_flaws()
    RobustnessAnalyzer->>RobustnessAnalyzer: compute_generalization_metrics()
    RobustnessAnalyzer->>PlotGenerator: plot_robustness_metrics(metrics)
    RobustnessAnalyzer-->>AnalysisRunner: robustness_insights, robustness_plots
    
    AnalysisRunner->>ComputationalAnalyzer: analyze(metrics_df)
    ComputationalAnalyzer->>ComputationalAnalyzer: compute_sample_efficiency()
    ComputationalAnalyzer->>ComputationalAnalyzer: compute_inference_times()
    ComputationalAnalyzer->>ComputationalAnalyzer: compute_memory_usage()
    ComputationalAnalyzer->>PlotGenerator: plot_computational_metrics(metrics)
    ComputationalAnalyzer-->>AnalysisRunner: computational_insights, computational_plots
    
    AnalysisRunner->>HyperparameterAnalyzer: analyze(metrics_df)
    HyperparameterAnalyzer->>HyperparameterAnalyzer: plot_win_rate_vs_learning_rate()
    HyperparameterAnalyzer->>HyperparameterAnalyzer: plot_win_rate_vs_discount_factor()
    HyperparameterAnalyzer->>HyperparameterAnalyzer: plot_win_rate_vs_exploration_rate()
    HyperparameterAnalyzer->>PlotGenerator: plot_hyperparameter_sensitivity(metrics)
    HyperparameterAnalyzer-->>AnalysisRunner: hyperparameter_insights, hyperparameter_plots
    
    AnalysisRunner->>ReportGenerator: generate_report(all_insights, all_plots)
    ReportGenerator->>ReportGenerator: create_comparative_table()
    ReportGenerator->>ReportGenerator: synthesize_findings()
    ReportGenerator->>PlotGenerator: generate_summary_visualizations()
    ReportGenerator->>FileSystem: write_report("final_report.pdf")
    ReportGenerator-->>AnalysisRunner: report_path
    
    AnalysisRunner-->>DataScientist: Analysis complete: report_path
```

### Code-Level Function Structure: AnalysisRunner

```python
class AnalysisRunner:
    def __init__(self, config: dict):
        self.config = config
        self.data_loader = DataLoader()
        self.analyzers = {
            'policy': PolicyAnalyzer(),
            'stability': StabilityAnalyzer(),
            'robustness': RobustnessAnalyzer(),
            'computational': ComputationalAnalyzer(),
            'hyperparameter': HyperparameterAnalyzer(),
        }
        self.plot_generator = PlotGenerator()
        self.report_generator = ReportGenerator()
    
    def run_analysis(self, metrics_dir: str):
        # Load data
        metrics_df = self.data_loader.load_metrics(metrics_dir)
        
        # Run 5-point analysis
        all_insights = {}
        all_plots = {}
        
        for analyzer_name, analyzer in self.analyzers.items():
            insights, plots = analyzer.analyze(metrics_df)
            all_insights[analyzer_name] = insights
            all_plots[analyzer_name] = plots
        
        # Generate report
        report_path = self.report_generator.generate_report(
            all_insights, all_plots, output_dir=self.config['output_dir']
        )
        
        return report_path

class PolicyAnalyzer:
    def analyze(self, metrics_df: pd.DataFrame) -> tuple[dict, list]:
        insights = {
            'aggression_score': self._compute_aggression_score(metrics_df),
            'defense_score': self._compute_defense_score(metrics_df),
            'efficiency_score': self._compute_efficiency_score(metrics_df),
        }
        
        plots = [
            self.plot_generator.plot_aggression_over_time(metrics_df),
            self.plot_generator.plot_defense_over_time(metrics_df),
            self.plot_generator.plot_efficiency_over_time(metrics_df),
        ]
        
        return insights, plots
```

---

## Code-Level Class Relationships

### Agent Interface Hierarchy

```python
# Abstract base class
class Agent(ABC):
    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def needs_replay_learning(self) -> bool:
        pass
    
    @abstractmethod
    def act(self, state: State) -> int:
        pass
    
    @abstractmethod
    def learn_from_replay(self, *args):
        pass
    
    @abstractmethod
    def learn_from_rollout(self, *args):
        pass
    
    @abstractmethod
    def push_to_replay_buffer(self, *args):
        pass

# Concrete implementations
class RandomAgent(Agent):
    is_on_policy = False
    needs_replay_learning = False

class TabularQAgent(Agent):
    is_on_policy = False
    needs_replay_learning = True

class TDAgent(Agent):
    is_on_policy = False
    needs_replay_learning = True

class DQNAgent(Agent):
    is_on_policy = False
    needs_replay_learning = True

class PPOAgent(Agent):
    is_on_policy = True
    needs_replay_learning = False

class MCTSAgent(Agent):
    is_on_policy = True
    needs_replay_learning = False
```

---

## Summary

This Level 4 code diagram shows:

1. **Detailed sequence diagrams** for key scenarios:
   - On-policy training (PPO)
   - Off-policy training (DQN)
   - State abstraction process
   - Reward shaping with strategy pattern
   - Metrics collection and export
   - 5-point analysis execution

2. **Code-level class structures** showing:
   - Class attributes and methods
   - Method signatures
   - Implementation details

3. **Function call flows** showing:
   - Exact method invocations
   - Parameter passing
   - Return values

These diagrams provide the granular detail needed to understand how the code actually executes and how components interact at the implementation level.

---

## Navigation

- [C4 Level 1: System Context](./c4-level1-context.md)
- [C4 Level 2: Container Diagram](./c4-level2-container.md)
- [C4 Level 3: Component Diagrams](./c4-level3-components.md)
- **C4 Level 4: Code Diagrams** (this document)

