"""
Agent Template

This file serves as a template and example for creating new agent types.
Copy this file and modify it to create your own agent.

To use this template:
1. Copy this file to a new file (e.g., my_agent.py)
2. Rename the class (e.g., MyAgent)
3. Implement the required methods
4. Register your agent with AgentRegistry
5. Use it in configuration files

See docs/EXTENDING_AGENTS.md for detailed instructions.
"""

from typing import Optional
from .base_agent import Agent
from ..utils.state import State


class AgentTemplate(Agent):
    """
    Template agent - replace this with your agent's description.
    
    This is a minimal example showing the required structure.
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """
        Initialize your agent.
        
        Args:
            seed: Random seed for reproducibility (optional)
            **kwargs: Additional parameters from config file
                     (accept these to be compatible with config system)
        
        Note:
            All parameters from the config file (except 'type') are passed
            as keyword arguments. Accept **kwargs to handle any extra params.
        """
        self.seed = seed
        if seed is not None:
            import random
            random.seed(seed)
        
        # Initialize your agent's internal state here
        # Example: self.my_param = kwargs.get('my_param', default_value)
    
    # ============================================================================
    # REQUIRED: These methods MUST be implemented
    # ============================================================================
    
    @property
    def is_on_policy(self) -> bool:
        """
        Whether this agent uses on-policy learning.
        
        Returns:
            True for on-policy agents (PPO, MCTS, policy gradients)
            False for off-policy agents (Q-Learning, DQN, value-based)
        
        Examples:
            - Q-Learning: False (off-policy)
            - DQN: False (off-policy)
            - PPO: True (on-policy)
            - Random: False (doesn't learn, but treated as off-policy)
        """
        return False  # Change based on your agent type
    
    @property
    def needs_replay_learning(self) -> bool:
        """
        Whether this agent requires replay buffer for learning.
        
        Returns:
            True if agent uses experience replay (Q-Learning, DQN)
            False otherwise (on-policy agents, non-learning agents)
        
        Examples:
            - Q-Learning: True (uses replay buffer)
            - DQN: True (uses replay buffer)
            - PPO: False (uses rollouts, not replay)
            - Random: False (doesn't learn)
        """
        return False  # Change based on your agent type
    
    def act(self, state: State) -> int:
        """
        Select an action given the current state.
        
        This is the core method - implement your action selection logic here.
        
        Args:
            state: Current State object containing:
                - state.valid_moves: List of valid action indices (e.g., [0, 1, 2])
                - state.player_pieces: List of 4 piece positions for your player
                - state.enemy_pieces: List of 3 lists, each with 4 enemy piece positions
                - state.dice_roll: Current dice value (1-6)
                - state.abstract_state: Hashable tuple for tabular methods
                - state.full_vector: NumPy array for neural networks
                - state.movable_pieces: List of piece indices that can move
        
        Returns:
            Action index (must be an element of state.valid_moves)
        
        Raises:
            ValueError: If no valid moves available (shouldn't happen, but handle gracefully)
        
        Example:
            # Simple: return first valid move
            return state.valid_moves[0]
            
            # Random: return random valid move
            import random
            return random.choice(state.valid_moves)
            
            # Greedy: return best move based on some scoring
            best_action = max(state.valid_moves, key=lambda a: self._score(state, a))
            return best_action
        """
        # TODO: Implement your action selection logic
        if not state.valid_moves:
            # Should not happen, but handle gracefully
            return 0
        
        # Example: return first valid move
        return state.valid_moves[0]
    
    # ============================================================================
    # OPTIONAL: Implement these if your agent needs them
    # ============================================================================
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """
        Learn from experience replay buffer.
        
        Called after push_to_replay_buffer() for off-policy agents.
        Implement this if your agent uses experience replay.
        
        Example (Q-Learning):
            if len(self.replay_buffer) < self.batch_size:
                return
            
            batch = self.replay_buffer.sample(self.batch_size)
            for state, action, reward, next_state, done in batch:
                # Update Q-values
                self.q_table[state][action] += self.learning_rate * (
                    reward + self.gamma * max(self.q_table[next_state]) - 
                    self.q_table[state][action]
                )
        """
        pass  # Override if needed
    
    def learn_from_rollout(self, rollout_buffer: list, *args, **kwargs) -> None:
        """
        Learn from a rollout buffer (on-policy learning).
        
        Called at the end of each episode for on-policy agents.
        Implement this if your agent uses on-policy learning (PPO, etc.).
        
        Args:
            rollout_buffer: List of experience dictionaries from the episode
                Each dict contains: 'state', 'action', 'reward', 'next_state', 'done', 'info'
        
        Example (PPO):
            states = [exp['state'] for exp in rollout_buffer]
            actions = [exp['action'] for exp in rollout_buffer]
            rewards = [exp['reward'] for exp in rollout_buffer]
            
            # Calculate advantages, update policy network
            advantages = self._compute_advantages(rewards)
            self._update_policy(states, actions, advantages)
        """
        pass  # Override if needed
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float,
                             next_state: State, done: bool, **kwargs) -> None:
        """
        Add experience to replay buffer.
        
        Called during training for off-policy agents.
        Implement this if your agent uses experience replay.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode terminated
            **kwargs: Additional info (may contain 'info' dict from environment)
        
        Example:
            experience = (state.abstract_state, action, reward, 
                         next_state.abstract_state, done)
            self.replay_buffer.append(experience)
        """
        pass  # Override if needed
    
    def on_episode_end(self) -> None:
        """
        Called at the end of each episode.
        
        Useful for:
        - Epsilon decay (exploration schedule)
        - Logging episode statistics
        - Resetting episode-specific state
        
        Example (epsilon decay):
            self.epsilon = max(self.min_epsilon, 
                             self.epsilon * self.epsilon_decay)
        """
        pass  # Override if needed
    
    def save(self, filepath: str) -> None:
        """
        Save agent model/parameters to file.
        
        Implement this if your agent has state that should be saved
        (e.g., neural network weights, Q-tables).
        
        Args:
            filepath: Path to save file
        
        Example:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': dict(self.q_table),
                    'epsilon': self.epsilon
                }, f)
        """
        pass  # Override if needed
    
    def load(self, filepath: str) -> None:
        """
        Load agent model/parameters from file.
        
        Implement this if your agent can be loaded from a saved state.
        
        Args:
            filepath: Path to load file from
        
        Example:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
        """
        pass  # Override if needed
    
    @property
    def supports_score_debug(self) -> bool:
        """
        Whether this agent exposes detailed scoring debug information.
        
        Return True if your agent can provide score breakdowns for debugging.
        If True, also implement get_last_score_debug().
        
        Returns:
            True if score debugging is supported, False otherwise
        """
        return False  # Override if you implement score debugging
    
    def get_last_score_debug(self) -> dict | None:
        """
        Get debug information for the last action selection.
        
        Only called if supports_score_debug returns True.
        Return a JSON-serializable dictionary with score components.
        
        Returns:
            Dictionary with debug info, or None if not available
        
        Example:
            return {
                'action': self.last_action,
                'scores': {
                    'action_0': 10.5,
                    'action_1': 8.2,
                    'action_2': 15.3
                },
                'total_score': 15.3,
                'decision_type': 'GREEDY'
            }
        """
        return None  # Override if you implement score debugging
    
    # ============================================================================
    # HELPER METHODS: Add your own helper methods below
    # ============================================================================
    
    def _score_action(self, state: State, action: int) -> float:
        """
        Example helper method: score an action.
        
        Add your own helper methods as needed.
        """
        # TODO: Implement your scoring logic
        return 0.0

