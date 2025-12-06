"""
Deep Q-Network (DQN) Agent for Ludo.

CPU-Optimized Implementation:
- Uses same state abstraction as TabularQAgent (combined, zone_based, or potential)
- Experience replay with efficient CPU batch processing
- Small neural network architecture optimized for CPU inference
- Target network for stable learning
- Optimized for 8 CPU cores with MKLDNN support

Based on TabularQAgent's state representation and reward structure.
"""

import random
import pickle
from typing import Dict, Tuple, Optional, List
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from rl_agent_ludo.utils.state import State
from rl_agent_ludo.agents.baseAgent import Agent
# --- Constants (Synced with LudoEnv) ---
# Localized to prevent ImportError if tabularQAgent is missing
POT_HOME = 0
POT_GOAL = 1
POT_SAFE = 2
POT_RISK_LOW = 3   # Risk in Q1/Q2
POT_RISK_HIGH = 4  # Risk in Q3/Q4
POT_KILL = 5
POT_DANGER = 6     # Static Danger
POT_Q1 = 7
POT_Q2 = 8
POT_Q3 = 9
POT_Q4 = 10

# Context
CTX_TRAILING = 0
CTX_NEUTRAL = 1
CTX_LEADING = 2

# Board Indices
HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]
HOME_CORRIDOR = list(range(52, 57))

# Static Danger Zones (Must match ludoEnv.py)
DANGER_INDEXES = [
    14, 15, 16, 17, 18, 19,  # After P2 spawn
    27, 28, 29, 30, 31, 32,  # After P3 spawn
    40, 41, 42, 43, 44, 45   # After P4 spawn
]


class DQNNetwork(nn.Module):
    """
    Small neural network for DQN optimized for CPU inference.
    
    Architecture:
    - Input: State tuple (size depends on abstraction: 9, 12, or 13)
    - Hidden layers: Configurable (default: [128, 128, 64] for combined state)
    - Output: 4 Q-values (one per piece)
    
    Optimized for CPU:
    - Small hidden layers to reduce computation
    - ReLU activations (fast on CPU)
    - Layer normalization instead of BatchNorm (works with batch size 1)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128, 64], output_dim: int = 4):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Use LayerNorm instead of BatchNorm (works with batch size 1)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (no activation, raw Q-values)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier initialization for better convergence)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, 4)
        """
        return self.network(x)


class DQNAgent(Agent):
    """
    Deep Q-Network agent for Ludo, optimized for CPU.
    
    Features:
    - Same state abstraction as TabularQAgent (potential, zone_based, combined)
    - Experience replay buffer
    - Target network for stable Q-learning
    - CPU-optimized batch processing
    - Uses rich reward structure from environment
    
    State Abstraction:
    - 'compact': 6-tuple (Context, P1, P2, P3, P4, Dice) - RECOMMENDED, matches TabularQAgent
    - 'potential': 9-tuple (P1-P4, Context, T1-T4)
    - 'zone_based': 12-tuple (HOME, PATH, SAFE, GOAL, EV1-EV4, TV1-TV4)
    - 'combined': 14-tuple (HOME, PATH, SAFE, GOAL, P1-P4, T1-T4, Context, Dice) - RECOMMENDED, matches TabularQAgent
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        batch_size: int = 128,  # Reduced batch size for faster training (was 256)
        replay_buffer_size: int = 10000,
        target_update_frequency: int = 100,
        train_frequency: int = 32,  # Train every N steps (increased to reduce overhead)
        gradient_steps: int = 1,  # Single gradient step per training call (reduced for speed)
        hidden_dims: List[int] = [128, 128, 64],
        state_abstraction: str = 'combined',  # 'potential', 'zone_based', or 'combined'
        device: Optional[str] = None,
        seed: Optional[int] = None,
        observation_mode: str = 'hybrid',  # 'hybrid' or 'hybrid_penalty' - syncs with Env
    ) -> None:
        """
        Initialize DQN agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum epsilon value
            batch_size: Batch size for training (larger = better CPU utilization)
            replay_buffer_size: Size of experience replay buffer
            target_update_frequency: Steps between target network updates
            train_frequency: Steps between training calls
            gradient_steps: Number of gradient steps per training call
            hidden_dims: Hidden layer dimensions
            state_abstraction: State abstraction method ('potential', 'zone_based', 'combined')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Device setup (force CPU for optimization)
        if device is None:
            device = 'cpu'  # Force CPU for this implementation
        self.device = torch.device(device)
        
        # Set PyTorch CPU optimizations
        if self.device.type == 'cpu':
            # Use fewer threads for better performance on small batches
            torch.set_num_threads(4)  # Reduced from 8 for better efficiency
            # Enable MKLDNN for better CPU performance
            torch.backends.mkldnn.enabled = True
            # Disable deterministic mode for speed
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False  # Not applicable to CPU
        
        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_frequency = target_update_frequency
        self.train_frequency = train_frequency
        self.gradient_steps = gradient_steps
        self.state_abstraction = state_abstraction
        self.observation_mode = observation_mode
        
        # --- INPUT DIMENSION CALCULATION (One-Hot Encoded) ---
        if state_abstraction == 'compact':
            # Context (3) + 4 * Potentials (11) + Dice (6) = 53
            # Potentials: HOME, GOAL, SAFE, RISK_LOW, RISK_HIGH, KILL, DANGER, Q1, Q2, Q3, Q4 (11 classes)
            self.input_dim = 3 + (11 * 4) + 6  # One-hot encoded
        elif state_abstraction == 'potential':
            input_dim = 9  # Legacy mode (not recommended)
        elif state_abstraction == 'zone_based':
            input_dim = 12  # Legacy
        elif state_abstraction in ['combined', 'enhanced']:
            # Combined: 14 raw values + 6 one-hot dice = 20
            # Raw: Home(1) + Path(1) + Safe(1) + Goal(1) + P1-P4(4) + T1-T4(4) + Context(1) + Dice(1) = 14
            # One-hot dice: 6 dimensions
            input_dim = 14 + 6  # 14 raw + 6 one-hot dice = 20
        else:
            input_dim = 53  # Default to compact one-hot size
        
        # Store raw input_dim for legacy modes, but use one-hot for compact
        if state_abstraction == 'compact':
            input_dim = self.input_dim
        elif state_abstraction in ['combined', 'enhanced']:
            input_dim = 20  # Use the calculated value
            self.input_dim = input_dim
        else:
            self.input_dim = input_dim
        
        # Neural networks
        self.q_network = DQNNetwork(input_dim, hidden_dims, output_dim=4).to(self.device)
        self.target_network = DQNNetwork(input_dim, hidden_dims, output_dim=4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer (Adam is good for CPU)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
        # Last state/action tracking (for replay buffer)
        self._last_state_tuple: Optional[Tuple] = None
        self._last_action: Optional[int] = None
        self._last_piece_idx: Optional[int] = None
        
        # Cache for state building (to avoid recomputation)
        self._state_cache: Dict[int, Tuple] = {}
        self._cache_size_limit = 1000  # Limit cache size
        
        # Reuse TabularQAgent's state building methods
        # We'll import the helper methods or recreate them
        self._init_state_builders()
    
    def _init_state_builders(self):
        """Initialize state building methods (reused from TabularQAgent logic)."""
        # These will be implemented to match TabularQAgent's state abstraction
        pass
    
    @property
    def is_on_policy(self) -> bool:
        """DQN is off-policy."""
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """DQN uses experience replay."""
        return True
    
    def act(self, state: State) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current state
            
        Returns:
            Action index (piece index that can move)
        """
        if not state.valid_moves:
            return 0
        
        # Filter valid moves to only include valid piece indices (0-3)
        # This ensures we only return actions that are actually movable pieces
        valid_actions = [a for a in state.valid_moves if isinstance(a, int) and 0 <= a < 4]
        
        if not valid_actions:
            # Fallback: if no valid actions, return first valid move (shouldn't happen)
            return state.valid_moves[0] if state.valid_moves else 0
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: use Q-network
        # Build One-Hot Tensor
        state_tensor = self._build_state_tensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]  # Shape: (4,)
        
        # Map valid actions to piece indices and select best
        piece_indices = [
            self._action_to_piece_idx(state, action) for action in valid_actions
        ]
        
        # Find best action
        best_action = valid_actions[0]
        best_q = q_values[piece_indices[0]]
        
        for action, piece_idx in zip(valid_actions, piece_indices):
            if q_values[piece_idx] > best_q:
                best_q = q_values[piece_idx]
                best_action = action
        
        # Store tuple for replay (we store the raw tuple to save RAM, encode on fly during train)
        self._last_state_tuple = self._build_state_tuple(state)
        self._last_action = best_action
        self._last_piece_idx = piece_indices[valid_actions.index(best_action)]
        
        return best_action
    
    def push_to_replay_buffer(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        **kwargs,
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Use stored state tuple if available (from act), otherwise build it
        if self._last_state_tuple is None:
            state_tuple = self._build_state_tuple(state)
            piece_idx = self._action_to_piece_idx(state, action)
        else:
            state_tuple = self._last_state_tuple
            piece_idx = self._last_piece_idx if self._last_piece_idx is not None else self._action_to_piece_idx(state, action)
        
        next_state_tuple = self._build_state_tuple(next_state)
        
        # Store experience (store piece_idx instead of action for easier training)
        self.replay_buffer.append((
            state_tuple,
            piece_idx,  # Store piece index for Q-value lookup
            reward,
            next_state_tuple,
            done,
        ))
        
        self.step_count += 1
        
        # Train periodically
        if len(self.replay_buffer) >= self.batch_size and self.step_count % self.train_frequency == 0:
            self.learn_from_replay()
        
        # Update target network periodically
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reset tracking
        if done:
            self._last_state_tuple = None
            self._last_action = None
            self._last_piece_idx = None
            self.on_episode_end()
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """
        Learn from experience replay buffer.
        
        Optimized for speed with single gradient step and batched tensor operations.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Perform gradient steps (optimized: single step is faster)
        for _ in range(self.gradient_steps):
            # Sample batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert raw tuples to Tensors on the fly
            if self.state_abstraction == 'compact':
                state_tensors = torch.cat([self._one_hot_encode(s) for s in states]).to(self.device)
                next_state_tensors = torch.cat([self._one_hot_encode(s) for s in next_states]).to(self.device)
            elif self.state_abstraction in ['combined', 'enhanced']:
                state_tensors = torch.cat([self._one_hot_encode_combined(s) for s in states]).to(self.device)
                next_state_tensors = torch.cat([self._one_hot_encode_combined(s) for s in next_states]).to(self.device)
            else:
                # Legacy modes: direct tensor conversion
                state_array = np.array(states, dtype=np.float32)
                next_state_array = np.array(next_states, dtype=np.float32)
                state_tensors = torch.from_numpy(state_array).to(self.device)
                next_state_tensors = torch.from_numpy(next_state_array).to(self.device)
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
            
            # Get Q-values for current states
            q_values = self.q_network(state_tensors)  # Shape: (batch_size, 4)
            
            # Get actions as piece indices (already stored as piece indices in replay buffer)
            action_piece_indices = torch.tensor(actions, dtype=torch.long).to(self.device)
            
            # Get Q-values for taken actions
            q_values_selected = q_values.gather(1, action_piece_indices.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values using target network
            with torch.no_grad():
                next_q_values = self.target_network(next_state_tensors)  # Shape: (batch_size, 4)
                # For next states, we need to mask invalid actions
                # For simplicity, take max (will be improved with action masking if needed)
                max_next_q = next_q_values.max(1)[0]
                target_q_values = rewards_tensor + (self.gamma * max_next_q * ~dones_tensor)
            
            # Compute loss
            loss = F.mse_loss(q_values_selected, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def learn_from_rollout(self, *args, **kwargs) -> None:
        """DQN doesn't use rollout learning."""
        pass
    
    def on_episode_end(self) -> None:
        """Update epsilon at end of episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        self.episode_count += 1
    
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'state_abstraction': self.state_abstraction,
            'replay_buffer': list(self.replay_buffer),  # Convert deque to list
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        # weights_only=False needed for PyTorch 2.6+ because checkpoints contain numpy objects
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        if 'state_abstraction' in checkpoint:
            self.state_abstraction = checkpoint['state_abstraction']
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.replay_buffer_size)
    
    # State building methods (reused from TabularQAgent)
    def _build_state_tuple(self, state: State) -> Tuple:
        """Build state tuple based on abstraction method."""
        if self.state_abstraction == 'compact':
            return self._build_compact_state(state)
        elif self.state_abstraction == 'zone_based':
            return self._build_zone_based_state(state)
        elif self.state_abstraction in ['combined', 'enhanced']:
            return self._build_combined_state(state)
        else:  # 'potential'
            return self._build_potential_state(state)
    
    def _build_potential_state(self, state: State) -> Tuple:
        """Build potential-based state tuple (9-tuple) - optimized."""
        potentials = []
        threat_flags = []
        player_pieces = state.player_pieces
        enemy_pieces = state.enemy_pieces
        
        # Batch process pieces for better performance
        for piece_idx in range(4):
            potentials.append(self._classify_potential(state, piece_idx))
            pos = player_pieces[piece_idx]
            # Optimized threat check
            is_threatened = self._is_token_under_threat(pos, enemy_pieces)
            threat_flags.append(1 if is_threatened else 0)
        
        context = self._compute_context(state)
        
        return (
            potentials[0], potentials[1], potentials[2], potentials[3],
            context,
            threat_flags[0], threat_flags[1], threat_flags[2], threat_flags[3]
        )
    
    def _build_compact_state(self, state: State) -> Tuple:
        """
        Build compact state tuple (6-tuple) - matches TabularQAgent's compact state.
        
        State Definition: (Context, Potential_1, Potential_2, Potential_3, Potential_4, Dice)
        
        - Context (3 values): Trailing(0), Neutral(1), Leading(2)
        - Potentials (11 values × 4 pieces): Quarter-based 11-class system with risk split
          0: HOME, 1: GOAL, 2: SAFE, 3: RISK_LOW, 4: RISK_HIGH, 5: KILL, 6: DANGER, 7: Q1, 8: Q2, 9: Q3, 10: Q4
        - Dice (6 values): Current dice roll (1-6, normalized to 0-5)
        
        This matches the TabularQAgent's compact state for consistency.
        """
        # 1. Game context (strategic situation)
        context = self._compute_context(state)
        
        # 2. Refined potentials (10-class system with quarter-based granularity and risk split)
        potentials = []
        for piece_idx in range(4):
            potential = self._classify_potential(state, piece_idx)
            potentials.append(potential)
        
        # 3. Dice roll (CRITICAL: explicitly included to solve "Blind Surgeon" problem)
        dice_roll = state.dice_roll
        # Normalize dice to 0-5 for state space (1-6 → 0-5)
        dice_normalized = dice_roll - 1
        
        return (
            context,
            potentials[0], potentials[1], potentials[2], potentials[3],
            dice_normalized
        )
    
    def _build_zone_based_state(self, state: State) -> Tuple:
        """Build zone-based state tuple (12-tuple)."""
        player_pieces = state.player_pieces
        
        home_count = sum(1 for pos in player_pieces if pos == HOME_INDEX)
        goal_count = sum(1 for pos in player_pieces if pos == GOAL_INDEX)
        safe_count = sum(1 for pos in player_pieces 
                        if pos not in [HOME_INDEX, GOAL_INDEX] 
                        and self._is_safe_position(pos))
        path_count = sum(1 for pos in player_pieces 
                        if 1 <= pos <= 51 
                        and not self._is_safe_position(pos))
        
        home_count = min(home_count, 4)
        path_count = min(max(path_count, 0), 4)
        safe_count = min(safe_count, 4)
        goal_count = min(goal_count, 4)
        
        ev_flags = []
        tv_flags = []
        
        for piece_idx in range(4):
            pos = player_pieces[piece_idx]
            if pos in [HOME_INDEX, GOAL_INDEX]:
                ev_flags.append(0)
                tv_flags.append(0)
                continue
            
            can_capture = False
            if self._is_piece_movable(state, piece_idx):
                next_pos = self._simulate_move(pos, state.dice_roll)
                can_capture = self._can_capture(next_pos, state.enemy_pieces)
            ev_flags.append(1 if can_capture else 0)
            
            is_threatened = self._is_token_under_threat(pos, state.enemy_pieces)
            tv_flags.append(1 if is_threatened else 0)
        
        return (
            home_count, path_count, safe_count, goal_count,
            ev_flags[0], ev_flags[1], ev_flags[2], ev_flags[3],
            tv_flags[0], tv_flags[1], tv_flags[2], tv_flags[3]
        )
    
    def _build_combined_state(self, state: State) -> Tuple:
        """
        Build combined state tuple (14-tuple) - optimized.
        
        State Definition: (Home, Path, Safe, Goal, P1, P2, P3, P4, T1, T2, T3, T4, Context, Dice)
        
        - Counts (4 values): Home, Path, Safe, Goal counts
        - Potentials (4 values): P1-P4 (0-10)
        - Threats (4 values): T1-T4 (0 or 1)
        - Context (1 value): 0-2
        - Dice (1 value): 0-5 (normalized from 1-6)
        
        CRITICAL: Dice is explicitly included to help network learn patterns like
        "dice=6 enables exiting home" vs "dice=1 doesn't", even when potentials are the same.
        """
        player_pieces = state.player_pieces
        enemy_pieces = state.enemy_pieces
        
        # Optimized counting with single pass
        home_count = 0
        goal_count = 0
        safe_count = 0
        path_count = 0
        
        for pos in player_pieces:
            if pos == HOME_INDEX:
                home_count += 1
            elif pos == GOAL_INDEX:
                goal_count += 1
            elif pos not in [HOME_INDEX, GOAL_INDEX] and self._is_safe_position(pos):
                safe_count += 1
            elif 1 <= pos <= 51:
                path_count += 1
        
        home_count = min(home_count, 4)
        path_count = min(max(path_count, 0), 4)
        safe_count = min(safe_count, 4)
        goal_count = min(goal_count, 4)
        
        potentials = []
        threat_flags = []
        
        # Batch process pieces
        for piece_idx in range(4):
            potentials.append(self._classify_potential(state, piece_idx))
            pos = player_pieces[piece_idx]
            is_threatened = self._is_token_under_threat(pos, enemy_pieces)
            threat_flags.append(1 if is_threatened else 0)
        
        context = self._compute_context(state)
        
        # Dice roll (CRITICAL: explicitly included to solve "Blind Surgeon" problem)
        dice_roll = state.dice_roll
        dice_normalized = dice_roll - 1  # Normalize 1-6 → 0-5
        
        return (
            home_count, path_count, safe_count, goal_count,
            potentials[0], potentials[1], potentials[2], potentials[3],
            threat_flags[0], threat_flags[1], threat_flags[2], threat_flags[3],
            context,
            dice_normalized
        )
    
    def _build_state_tensor(self, state: State) -> torch.Tensor:
        """Builds One-Hot Tensor directly from State."""
        tuple_rep = self._build_state_tuple(state)
        if self.state_abstraction == 'compact':
            return self._one_hot_encode(tuple_rep)
        elif self.state_abstraction in ['combined', 'enhanced']:
            return self._one_hot_encode_combined(tuple_rep)
        else:
            # Legacy modes: direct tensor conversion
            return torch.tensor([tuple_rep], dtype=torch.float32)
    
    def _one_hot_encode(self, state_tuple: Tuple) -> torch.Tensor:
        """
        Converts (Context, P1, P2, P3, P4, Dice) -> One-Hot Vector (Size 53).
        
        Args:
            state_tuple: Raw state tuple (Context, P1, P2, P3, P4, Dice)
            
        Returns:
            One-hot encoded tensor of shape (1, 53)
        """
        if len(state_tuple) != 6:
            return torch.zeros(1, self.input_dim)
        
        ctx, p1, p2, p3, p4, dice = state_tuple
        
        # One-hot vectors (pre-allocate full vector for efficiency)
        vec = np.zeros(53, dtype=np.float32)
        
        # Context (indices 0-2)
        vec[int(ctx)] = 1.0
        
        # Potentials (indices 3-46, 11 each)
        vec[3 + int(p1)] = 1.0      # P1 at index 3-13
        vec[14 + int(p2)] = 1.0     # P2 at index 14-24
        vec[25 + int(p3)] = 1.0     # P3 at index 25-35
        vec[36 + int(p4)] = 1.0     # P4 at index 36-46
        
        # Dice (indices 47-52)
        vec[47 + int(dice)] = 1.0
        
        # Convert to tensor efficiently
        return torch.from_numpy(vec).unsqueeze(0)
    
    def _one_hot_encode_combined(self, state_tuple: Tuple) -> torch.Tensor:
        """
        Converts Combined State Tuple -> Tensor with One-Hot Dice.
        
        Args:
            state_tuple: Raw state tuple (Home, Path, Safe, Goal, P1, P2, P3, P4, T1, T2, T3, T4, Context, Dice)
            
        Returns:
            Tensor of shape (1, 20): 14 raw values + 6 one-hot dice
        """
        if len(state_tuple) != 14:
            return torch.zeros(1, self.input_dim)
        
        home, path, safe, goal, p1, p2, p3, p4, t1, t2, t3, t4, ctx, dice = state_tuple
        
        # Raw values (14 dimensions)
        raw_vec = np.array([
            float(home), float(path), float(safe), float(goal),
            float(p1), float(p2), float(p3), float(p4),
            float(t1), float(t2), float(t3), float(t4),
            float(ctx), float(dice)
        ], dtype=np.float32)
        
        # One-hot dice (6 dimensions)
        dice_one_hot = np.zeros(6, dtype=np.float32)
        dice_one_hot[int(dice)] = 1.0
        
        # Combine: 14 raw + 6 one-hot = 20 dimensions
        combined_vec = np.concatenate([raw_vec, dice_one_hot])
        
        return torch.from_numpy(combined_vec).unsqueeze(0)
    
    def _state_tuple_to_tensor(self, state_tuple: Tuple) -> torch.Tensor:
        """Convert state tuple to tensor (optimized)."""
        # Direct conversion is faster than going through list
        if isinstance(state_tuple, torch.Tensor):
            return state_tuple
        return torch.tensor(state_tuple, dtype=torch.float32)
    
    def _action_to_piece_idx(self, state: State, action_idx: int) -> int:
        """
        Map action to piece index.
        
        CRITICAL: We use ABSOLUTE indexing - action_idx IS the piece index.
        Action 0 = Token 0, Action 1 = Token 1, etc.
        This matches the environment's absolute indexing.
        """
        # With absolute indexing, action_idx is already the piece index
        return action_idx
    
    
    # --- HELPER LOGIC (Synced with LudoEnv) ---
    def _classify_potential(self, state: State, piece_idx: int) -> int:
        """
        Classify potential for one piece using 11-class quarter-based system with risk split.
        CRITICAL: Matches LudoEnv logic exactly - uses current_pos for Risk/Danger/Quarter,
        only uses next_pos for Kill (future event).
        """
        current_pos = state.player_pieces[piece_idx]
        dice = state.dice_roll

        # 1. Static Safe States (Current Position)
        if current_pos == HOME_INDEX:
            return POT_HOME
        if current_pos == GOAL_INDEX:
            return POT_GOAL
        if current_pos in GLOBE_INDEXES or current_pos in STAR_INDEXES:
            return POT_SAFE

        # 2. Kill (Future Event) - This remains lookahead
        if self._is_piece_movable(state, piece_idx):
            next_pos = self._simulate_move(current_pos, dice)
            if self._can_capture(next_pos, state.enemy_pieces):
                return POT_KILL

        # 3. Risk (Current State) - Synced with LudoEnv
        if self._is_token_under_threat(current_pos, state.enemy_pieces):
            if current_pos >= 27:
                return POT_RISK_HIGH
            else:
                return POT_RISK_LOW

        # 4. Danger (Static) - Only in experimental mode
        if self.observation_mode == 'hybrid_penalty':
            if current_pos in DANGER_INDEXES:
                return POT_DANGER

        # 5. Neutral Progress (Current Position)
        if 1 <= current_pos <= 13:
            return POT_Q1
        elif 14 <= current_pos <= 26:
            return POT_Q2
        elif 27 <= current_pos <= 39:
            return POT_Q3
        else:
            return POT_Q4
    
    def _is_piece_movable(self, state: State, piece_idx: int) -> bool:
        """Check if piece is movable."""
        if not state.valid_moves:
            return False
        if state.movable_pieces:
            return piece_idx in state.movable_pieces
        return piece_idx in state.valid_moves
    
    def _simulate_move(self, current_pos: int, dice_roll: int) -> int:
        """Simulate a move - synced with LudoEnv."""
        if current_pos == HOME_INDEX:
            if dice_roll == 6:
                return 1  # Standard ludo rule
            return HOME_INDEX
        if current_pos == GOAL_INDEX:
            return GOAL_INDEX
        
        next_pos = current_pos + dice_roll
        # Star jump logic
        if next_pos in STAR_INDEXES:
            idx = STAR_INDEXES.index(next_pos)
            if idx < len(STAR_INDEXES) - 1:
                next_pos = STAR_INDEXES[idx + 1]
            else:
                next_pos = GOAL_INDEX
        
        if next_pos > GOAL_INDEX:
            return GOAL_INDEX
        return next_pos
    
    def _is_at_goal(self, current_pos: int, dice_roll: int) -> bool:
        """Check if move reaches goal."""
        if current_pos == HOME_INDEX:
            return False
        if current_pos in HOME_CORRIDOR:
            return current_pos + dice_roll >= GOAL_INDEX
        if 1 <= current_pos <= 51:
            return current_pos + dice_roll >= GOAL_INDEX
        return False
    
    def _can_capture(self, next_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """Check if position can capture enemy - synced with LudoEnv."""
        # Cannot capture on safe globes/stars
        if next_pos in GLOBE_INDEXES or next_pos in STAR_INDEXES:
            return False
        if next_pos > 51:
            return False  # Safe corridor
        
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == next_pos:
                    return True
        return False
    
    def _is_safe_position(self, pos: int) -> bool:
        """Check if position is safe."""
        return (
            pos == HOME_INDEX
            or pos == GOAL_INDEX
            or pos in HOME_CORRIDOR
            or pos in GLOBE_INDEXES
            or pos == 1
        )
    
    def _is_star_jump(self, current_pos: int, next_pos: int, dice_roll: int) -> bool:
        """Check if move is a star jump."""
        if current_pos == HOME_INDEX or not (1 <= current_pos <= 51):
            return False
        original_landing = current_pos + dice_roll
        return original_landing in STAR_INDEXES and original_landing != next_pos
    
    def _get_circular_distance(self, pos1: int, pos2: int) -> int:
        """Get circular distance between positions (optimized)."""
        # Fast path for common case
        if pos1 in [HOME_INDEX, GOAL_INDEX] or pos2 in [HOME_INDEX, GOAL_INDEX]:
            return 100
        if not (1 <= pos1 <= 51) or not (1 <= pos2 <= 51):
            return 100
        # Optimized distance calculation
        diff = abs(pos1 - pos2)
        # Use min directly without extra checks
        return min(diff, 51 - diff)
    
    def _is_token_under_threat(self, token_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """Check if token is under threat - synced with LudoEnv."""
        if token_pos in GLOBE_INDEXES or token_pos in STAR_INDEXES or token_pos == HOME_INDEX or token_pos == GOAL_INDEX or token_pos > 51:
            return False
        
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == HOME_INDEX or e_pos == GOAL_INDEX:
                    continue
                # Simple distance check (approximate) - matches LudoEnv
                dist = (token_pos - e_pos) % 52
                # If enemy is 1-6 steps behind us
                if 1 <= (-dist % 52) <= 6:
                    return True
        return False
    
    def _calculate_probability_of_risk(
        self,
        next_pos: int,
        enemy_pieces: List[List[int]],
    ) -> float:
        """Calculate risk probability (optimized)."""
        if self._is_safe_position(next_pos):
            return 0.0
        risk_score = 0.0
        # Early exit optimization
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos in [HOME_INDEX, GOAL_INDEX] or self._is_safe_position(e_pos):
                    continue
                dist = self._get_circular_distance(next_pos, e_pos)
                if 1 <= dist <= 6:
                    risk_score += (7 - dist) * 800.0
                    # Early exit if risk is already high
                    if risk_score > 2000.0:
                        return risk_score
        return risk_score
    
    def _get_weighted_equity_score(self, pieces: List[int]) -> float:
        """Get weighted equity score."""
        score = 0.0
        for pos in pieces:
            if pos == GOAL_INDEX:
                score += 100
            elif pos in HOME_CORRIDOR:
                score += 50 + pos
            elif pos in GLOBE_INDEXES or pos == 1:
                score += 10 + pos
            elif pos == HOME_INDEX:
                score += 0
            else:
                score += pos
        return score
    
    def _compute_context(self, state: State) -> int:
        """Compute game context (optimized)."""
        my_score = self._get_weighted_equity_score(state.player_pieces)
        if state.enemy_pieces:
            # Optimized: compute max in one pass
            max_opp = max(self._get_weighted_equity_score(enemy) for enemy in state.enemy_pieces)
        else:
            max_opp = 0.0
        gap = my_score - max_opp
        if gap < -20:
            return CTX_TRAILING
        elif gap > 20:
            return CTX_LEADING
        return CTX_NEUTRAL

