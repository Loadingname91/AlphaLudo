"""
Trajectory Buffer for collecting full game trajectories.

This buffer stores complete game trajectories (state sequences) and labels them
with temporal discounting based on the final game outcome.

Purpose:
- Collect data for training the Win Probability Network (Coach)
- Store (State, Outcome) pairs with temporal discounting
- Support Phase 2: Data Infrastructure (Silent Watcher)
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import numpy as np


class TrajectoryBuffer:
    """
    Buffer for storing game trajectories with quadratic temporal discounting.

    Each trajectory is a sequence of states visited during a game.
    When the game ends, all states are labeled with quadratic temporal discounting.

    Quadratic Temporal Discounting Formula:
    Label_t = 0.5 + (Outcome - 0.5) * (t/T)^2

    Where:
    - Outcome: 1.0 for win, 0.0 for loss
    - T: Final timestep (game length - 1, 0-indexed)
    - t: Current timestep
    - (t/T)^2: Quadratic progress that accelerates near game end

    This ensures:
    - Early states (t=0-50): Labels stay near 0.5 (uncertainty)
    - Late states (t=80-100): Labels strongly reflect outcome
    - Better learning signal with label std ~0.38-0.42
    """
    
    def __init__(self, max_size: int = 50000, gamma: float = None):
        """
        Initialize trajectory buffer.

        Args:
            max_size: Maximum number of (state, label) pairs to store
            gamma: (Deprecated) No longer used with quadratic discounting. Kept for backward compatibility.
        """
        self.max_size = max_size
        # gamma is deprecated but kept for backward compatibility and metadata storage
        if gamma is not None:
            import warnings
            warnings.warn("gamma parameter is deprecated and no longer used with quadratic discounting",
                          DeprecationWarning, stacklevel=2)
        self.gamma = gamma  # Store for metadata purposes (even though deprecated for discounting)
        self.buffer = deque(maxlen=max_size)
        
        # Current trajectory being built
        self.current_trajectory: List[np.ndarray] = []
        
        # Statistics
        self.total_trajectories = 0
        self.total_states = 0
    
    def add_state(self, state_vector: np.ndarray) -> None:
        """
        Add a state to the current trajectory.
        
        Args:
            state_vector: State feature vector (expanded with relative features)
        """
        self.current_trajectory.append(state_vector.copy())
    
    def finalize_trajectory(self, outcome: float) -> int:
        """
        Finalize the current trajectory with quadratic temporal discounting.

        Uses quadratic progress weighting to give more importance to late-game states:
        - Early states (t=0-50) stay near 0.5 (uncertainty)
        - Late states (t=80-100) strongly reflect outcome
        - Increases label std to ~0.38-0.42 (better learning signal)

        Args:
            outcome: Final game outcome (1.0 for win, 0.0 for loss)

        Returns:
            Number of (state, label) pairs added to buffer
        """
        if len(self.current_trajectory) == 0:
            return 0

        T = len(self.current_trajectory) - 1  # Final timestep (0-indexed)
        num_added = 0

        # Label each state with quadratic temporal discounting
        for t, state in enumerate(self.current_trajectory):
            # Quadratic progress: (t/T)^2 gives more weight to late-game states
            progress = (t / T) ** 2  # Range: 0 to 1, accelerates near end

            # Blend outcome with 0.5 baseline using quadratic progress
            label = 0.5 + (outcome - 0.5) * progress

            # Win game: 0.5 → 0.6 → 0.8 → 0.95 → 1.0
            # Loss game: 0.5 → 0.4 → 0.2 → 0.05 → 0.0

            # Add to buffer
            self.buffer.append((state, label))
            num_added += 1

        # Clear current trajectory
        self.current_trajectory = []
        self.total_trajectories += 1
        self.total_states += num_added

        return num_added
    
    def get_dataset(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get dataset of (states, labels) for training.
        
        Args:
            max_samples: Maximum number of samples to return (None = all)
            
        Returns:
            Tuple of (states, labels) as numpy arrays
        """
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        samples = list(self.buffer)
        if max_samples is not None and len(samples) > max_samples:
            # Randomly sample if buffer is larger than requested
            indices = np.random.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in indices]
        
        states = np.array([s[0] for s in samples])
        labels = np.array([s[1] for s in samples])
        
        return states, labels
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.current_trajectory = []
        self.total_trajectories = 0
        self.total_states = 0
    
    def __len__(self) -> int:
        """Get current buffer size (supports len() builtin)."""
        return len(self.buffer)
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'buffer_size': len(self.buffer),
            'total_trajectories': self.total_trajectories,
            'total_states': self.total_states,
            'current_trajectory_length': len(self.current_trajectory),
        }

