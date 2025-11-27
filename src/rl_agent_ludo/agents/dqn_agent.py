"""
Dueling Double DQN Agent for Ludo.

Uses Dueling DQN architecture with Double Q-Learning, Prioritized Experience Replay,
and orthogonal state abstraction.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional
from .base_agent import Agent
from ..utils.state import State
from ..utils.orthogonal_state_abstractor import OrthogonalStateAbstractor
from .modules.dueling_dqn_network import DuelingDQNNetwork
from ..utils.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNAgent(Agent):
    """Dueling Double DQN with Prioritized Experience Replay."""
    
    def __init__(
        self,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        discount_factor: float = None,  # Alias for gamma (for config compatibility)
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 80000,
        target_update_freq: int = 1000,
        device: str = "cpu",
        # PER parameters
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_epsilon: float = 1e-6,
        # Ignore extra config keys
        seed: int = None,
        player_id: int = 0,
        debug_scores: bool = False,
        **kwargs  # Accept any extra params to prevent errors
    ):
        super().__init__()
        
        # Handle discount_factor alias
        if discount_factor is not None:
            gamma = discount_factor
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Debugging
        self.debug_scores = debug_scores
        self._last_score_debug = None
        self._last_td_error = None
        self._last_loss = None
        
        # State abstraction
        self.abstractor = OrthogonalStateAbstractor()
        
        # Networks
        self.online_net = DuelingDQNNetwork().to(self.device)
        self.target_net = DuelingDQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # Prioritized experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=int(buffer_size),
            alpha=float(per_alpha),
            beta=float(per_beta_start),
            beta_end=float(per_beta_end),
            epsilon=float(per_epsilon)
        )
    
    def act(self, state: State) -> int:
        """Select action using epsilon-greedy policy with action masking."""
        valid_actions = state.valid_moves
        decision_type = "UNKNOWN"
        selected_q = 0.0
        q_values_all = None
        
        if np.random.random() < self.epsilon:
            # Exploration: random valid action
            action = np.random.choice(valid_actions)
            decision_type = "EXPLORATION"
            
            # Still compute Q-values for debugging
            if self.debug_scores:
                phi_s = self.abstractor.get_orthogonal_state(state)
                phi_s_tensor = torch.FloatTensor(phi_s).to(self.device)
                with torch.no_grad():
                    q_values_all = self.online_net(phi_s_tensor).cpu().numpy()
                selected_q = float(q_values_all[action])
        else:
            # Exploitation: greedy action with masking
            decision_type = "EXPLOITATION"
            phi_s = self.abstractor.get_orthogonal_state(state)
            phi_s_tensor = torch.FloatTensor(phi_s).to(self.device)
            
            with torch.no_grad():
                q_values_all = self.online_net(phi_s_tensor).cpu().numpy()
            
            # Mask invalid actions
            masked_q = np.full(4, -np.inf)
            for action in valid_actions:
                masked_q[action] = q_values_all[action]
            
            action = np.argmax(masked_q)
            selected_q = float(q_values_all[action])
        
        # Store debug information if enabled
        if self.debug_scores:
            components = {}
            for i in range(4):
                if q_values_all is not None:
                    components[f'q_piece_{i}'] = float(q_values_all[i])
                else:
                    components[f'q_piece_{i}'] = 0.0
            
            self._last_score_debug = {
                'action': int(action),
                'dice_roll': int(state.dice_roll),
                'total_score': float(selected_q),
                'components': components,
                'decision_type': decision_type,
                'epsilon': float(self.epsilon),
                'valid_actions': [int(a) for a in valid_actions],
                'buffer_size': int(self.replay_buffer.size),
                'step_count': int(self.step_count)
            }
        
        return action
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float,
                             next_state: State, done: bool, **kwargs):
        """Add experience to prioritized replay buffer."""
        # Convert states to feature vectors
        phi_s = self.abstractor.get_orthogonal_state(state)
        phi_s_next = self.abstractor.get_orthogonal_state(next_state)
        
        # Add to buffer (priority will be set on first sample)
        self.replay_buffer.add(phi_s, action, reward, phi_s_next, done)
    
    def learn_from_replay(self, *args, **kwargs):
        """Learn from replay buffer using Double DQN with prioritized sampling."""
        if self.replay_buffer.size < self.batch_size:
            return
        
        # Sample batch
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch - convert to numpy arrays first for efficiency
        states_list = [b[0] for b in batch]
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states_list = [b[3] for b in batch]
        next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device)
        dones = torch.BoolTensor([b[4] for b in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        q_values = self.online_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
            # Double DQN: select action with online net, evaluate with target net
            with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(1)
            next_q_target = self.target_net(next_states)
            next_q_selected = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            targets = rewards + (self.gamma * next_q_selected * ~dones)
        
        # Calculate TD-errors for priority updates
        td_errors = (targets - q_selected).abs().detach().cpu().numpy()

        # Huber loss with importance sampling weights
        loss = torch.nn.functional.smooth_l1_loss(q_selected, targets, reduction='none')
        weighted_loss = (weights * loss).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Store debug info if enabled
        if self.debug_scores:
            self._last_td_error = float(td_errors.mean())
            self._last_loss = float(weighted_loss.item())
            if self._last_score_debug is not None:
                self._last_score_debug['td_error'] = self._last_td_error
                self._last_score_debug['loss'] = self._last_loss
                self._last_score_debug['buffer_size'] = int(self.replay_buffer.size)

        self.step_count += 1

        # Periodically update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            if self.debug_scores and self._last_score_debug is not None:
                self._last_score_debug['target_network_updated'] = True
    
    def on_episode_end(self):
        """Called at the end of each episode."""
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    # ---- Optional score debugging hooks ----------------------------------------
    
    @property
    def supports_score_debug(self) -> bool:
        """Indicate that this agent can provide score breakdowns when enabled."""
        return self.debug_scores
    
    def get_last_score_debug(self) -> Optional[dict]:
        """Return debug information for the last decision."""
        return self._last_score_debug
    
    def save(self, filepath: str):
        """Save agent model/parameters to file."""
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent model/parameters from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        return True

