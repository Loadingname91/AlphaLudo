import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network for Ludo Game.
    
    Architecture:
    Input (31,) -> Linear(128) -> LayerNorm -> ReLU
                -> Value Stream (128->1)
                -> Advantage Stream (128->4)
                -> Combine to Q(s,a)
    """
    def __init__(self, input_dim: int = 31, hidden_dim: int = 128, num_actions: int = 4):
        """
        Initialize the Dueling DQN Network.
        Args :
            input_dim (int): Dimension of the input state vector (default: 31)
            hidden_dim (int): Dimension of the hidden layer (default: 128)
            num_actions (int): Number of possible actions (default: 4)
        """
        super(DuelingDQNNetwork,self).__init__()

        # Shared Layers
        self.shared = nn.Linear(input_dim,hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Value Stream  How good is the current state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1) # outputs Single Value for the state
        )

        # Adavantage Stream : estimates A(s,a) - how much better is each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,num_actions) # outputs 4 values ( one per action)
        )
    
    def forward(self,state:torch.Tensor) -> torch.Tensor:
        """
        Forward pass : Input -> Shared -> Value / Advantage -> Combine to Q(s,a)
        Args :
            state (torch.Tensor): Input state vector of shape (31,)
        Returns :
            Q(s,a) (torch.Tensor): Output Q-values for all actions of shape (num_actions,)
        """
        # Store original input shape to handle 1D vs 2D inputs
        was_1d = state.dim() == 1
        
        # Ensure input is 2D (batch_size, features) for processing
        if was_1d:
            state = state.unsqueeze(0)  # (31,) -> (1, 31)
        
        # Shared Layers
        x = self.shared(state)  # (batch_size, 128)
        x = self.layer_norm(x)  # (batch_size, 128)
        x = F.relu(x)           # (batch_size, 128)
    
        # Split into value and advantage streams
        value = self.value_stream(x)      # (batch_size, 1)
        advantage = self.advantage_stream(x)  # (batch_size, 4)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This ensures identifiability (can't add constant to both)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))  # (batch_size, 4)

        # Remove batch dimension if original input was 1D
        if was_1d:
            q_values = q_values.squeeze(0)  # (1, 4) -> (4,)

        return q_values

    
