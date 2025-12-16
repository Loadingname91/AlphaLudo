"""
Reward Network for T-REX.

Learns to assign rewards to states such that better trajectories
get higher predicted returns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class RewardNetwork(nn.Module):
    """
    Neural network that predicts scalar reward for each state.

    The network is trained to assign rewards such that:
    - Better trajectories get higher cumulative rewards
    - Worse trajectories get lower cumulative rewards
    """

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Scalar reward output
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar reward for given state.

        Args:
            state: (batch_size, state_dim) or (state_dim,)

        Returns:
            reward: (batch_size, 1) or (1,)
        """
        return self.network(state)

    def predict_return(self, states: torch.Tensor,
                      discount: float = 0.99) -> torch.Tensor:
        """
        Predict discounted return for trajectory.

        Args:
            states: (trajectory_length, state_dim)
            discount: gamma discount factor

        Returns:
            total_return: scalar
        """
        rewards = self.forward(states)  # (T, 1)

        # Apply discount
        T = rewards.shape[0]
        discounts = torch.pow(discount, torch.arange(T, device=rewards.device)).unsqueeze(1)

        discounted_return = (rewards * discounts).sum()
        return discounted_return


class RewardLearner:
    """
    Trains reward network from preference pairs using Bradley-Terry model.
    """

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-5,
                 device: str = 'cpu'):
        """
        Initialize reward learner.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.reward_net = RewardNetwork(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(
            self.reward_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def ranking_loss(self, traj_better: Dict, traj_worse: Dict) -> Tuple[torch.Tensor, bool]:
        """
        Bradley-Terry ranking loss.

        P(better > worse) = exp(r_better) / (exp(r_better) + exp(r_worse))
        Loss = -log P(better > worse)
             = log(1 + exp(r_worse - r_better))

        Args:
            traj_better: Better trajectory dict
            traj_worse: Worse trajectory dict

        Returns:
            loss: scalar
            correct: True if predicted ranking is correct
        """
        # Convert states to tensors
        states_better = torch.FloatTensor(np.array(traj_better['states'])).to(self.device)
        states_worse = torch.FloatTensor(np.array(traj_worse['states'])).to(self.device)

        # Predict returns
        r_better = self.reward_net.predict_return(states_better)
        r_worse = self.reward_net.predict_return(states_worse)

        # Ranking loss (numerically stable)
        loss = torch.log(1 + torch.exp(r_worse - r_better))

        # Check if prediction is correct
        correct = (r_better > r_worse).item()

        return loss, correct

    def train_epoch(self, preference_pairs: List[Tuple], batch_size: int = 32) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            preference_pairs: List of (better, worse) trajectory pairs
            batch_size: Batch size

        Returns:
            avg_loss, avg_accuracy
        """
        self.reward_net.train()

        # Shuffle pairs
        import random
        pairs_shuffled = preference_pairs.copy()
        random.shuffle(pairs_shuffled)

        epoch_losses = []
        epoch_correct = []

        # Mini-batch training
        for i in range(0, len(pairs_shuffled), batch_size):
            batch = pairs_shuffled[i:i+batch_size]

            batch_loss = 0
            batch_correct = 0

            for traj_better, traj_worse in batch:
                loss, correct = self.ranking_loss(traj_better, traj_worse)
                batch_loss += loss
                batch_correct += int(correct)

            batch_loss = batch_loss / len(batch)

            # Backprop
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(batch_loss.item())
            epoch_correct.append(batch_correct / len(batch))

        return np.mean(epoch_losses), np.mean(epoch_correct)

    def validate(self, preference_pairs: List[Tuple]) -> Tuple[float, float]:
        """
        Compute validation loss and accuracy.

        Args:
            preference_pairs: Validation pairs

        Returns:
            avg_loss, avg_accuracy
        """
        self.reward_net.eval()

        val_losses = []
        val_correct = []

        with torch.no_grad():
            for traj_better, traj_worse in preference_pairs:
                loss, correct = self.ranking_loss(traj_better, traj_worse)
                val_losses.append(loss.item())
                val_correct.append(int(correct))

        return np.mean(val_losses), np.mean(val_correct)

    def train(self, train_pairs: List[Tuple], val_pairs: List[Tuple],
             num_epochs: int = 100, batch_size: int = 32,
             patience: int = 10, verbose: bool = True):
        """
        Full training loop with early stopping.

        Args:
            train_pairs: Training preference pairs
            val_pairs: Validation preference pairs
            num_epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training reward network for {num_epochs} epochs...")
            print(f"{'='*70}")
            print(f"Train pairs: {len(train_pairs)}")
            print(f"Val pairs: {len(val_pairs)}")
            print(f"Device: {self.device}")
            print(f"Batch size: {batch_size}\n")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_pairs, batch_size)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_pairs)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint('checkpoints/level6/reward_network_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self._load_checkpoint('checkpoints/level6/reward_network_best.pth')

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ Training complete!")
            print(f"{'='*70}")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"Final val accuracy: {self.val_accuracies[-1]:.3f}")

    def _save_checkpoint(self, filepath: str):
        """Save checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.reward_net.state_dict(), filepath)

    def _load_checkpoint(self, filepath: str):
        """Load checkpoint."""
        self.reward_net.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )

    def save(self, filepath: str):
        """Save full training state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'state_dict': self.reward_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, filepath)

        print(f"Reward learner saved to {filepath}")

    def load(self, filepath: str):
        """Load full training state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.reward_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']

        print(f"Reward learner loaded from {filepath}")
