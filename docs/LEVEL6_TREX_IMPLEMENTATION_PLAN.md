# Level 6: T-REX Implementation Plan for Ludo

**Goal**: Learn a reward function from ranked game trajectories, then train a policy that exceeds demonstrator performance.

**Timeline**: 3-4 weeks
**Difficulty**: Advanced (builds on existing curriculum)

---

## Table of Contents

1. [Overview](#overview)
2. [T-REX Algorithm Explained](#trex-algorithm-explained)
3. [Implementation Architecture](#implementation-architecture)
4. [Phase-by-Phase Roadmap](#phase-by-phase-roadmap)
5. [Code Structure](#code-structure)
6. [Evaluation Plan](#evaluation-plan)
7. [Expected Results](#expected-results)

---

## Overview

### What is T-REX?

**T-REX (Trajectory-ranked Reward EXtrapolation)** learns reward functions from preference rankings over trajectories, then trains policies using the learned reward.

**Key Insight**: You don't need optimal demonstrations - just rankings (e.g., "trajectory A was better than B").

### Why T-REX for Ludo?

1. ✅ **Leverage Existing Agents**: Use Level 1-5 agents as trajectory generators
2. ✅ **Natural Rankings**: Win/loss outcomes provide clear preferences
3. ✅ **Exceed Demonstrators**: Can learn strategies better than any single agent
4. ✅ **Handles Stochasticity**: Learns from outcomes, not individual moves
5. ✅ **Interpretable Rewards**: Can visualize what the network considers "good"

### High-Level Pipeline

```
┌─────────────────┐
│  Phase 1:       │
│  Collect        │──┐
│  Trajectories   │  │
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  Phase 2:       │  │
│  Rank           │◄─┘
│  Trajectories   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Phase 3:       │
│  Learn Reward   │
│  Function       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Phase 4:       │
│  Train Policy   │
│  with Learned   │
│  Reward         │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Phase 5:       │
│  Evaluate       │
│  vs Baseline    │
└─────────────────┘
```

---

## T-REX Algorithm Explained

### Step 1: Collect Trajectories

Run existing agents (Level 1-5) to collect full game trajectories:

```python
trajectory = {
    'states': [s_0, s_1, ..., s_T],      # State sequence
    'actions': [a_0, a_1, ..., a_T],     # Action sequence
    'rewards': [r_0, r_1, ..., r_T],     # Environment rewards (ignore these!)
    'outcome': 'win' or 'loss',           # Game result
    'final_score': 61,                    # Final win rate or score
    'num_captures': 5,                    # Additional metrics
    'episode_length': 120                 # Steps to completion
}
```

### Step 2: Create Preference Pairs

Rank trajectories by quality:

```python
# Ranking criteria (in order of importance):
# 1. Win > Loss
# 2. Among wins: more captures > fewer captures
# 3. Among wins: shorter > longer
# 4. Among losses: survived longer > died early

preference_pairs = []
for traj_i in trajectories:
    for traj_j in trajectories:
        if is_better(traj_i, traj_j):
            preference_pairs.append((traj_i, traj_j))  # i > j
```

### Step 3: Learn Reward Network

Train a neural network to predict trajectory returns such that better trajectories get higher predicted returns:

```python
class RewardNetwork(nn.Module):
    """Predicts cumulative return of a trajectory."""

    def __init__(self, state_dim=16, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Scalar reward for each state
        )

    def forward(self, state):
        """Returns scalar reward for given state."""
        return self.network(state)

    def predict_return(self, trajectory):
        """Sum rewards over trajectory."""
        total = 0
        for state in trajectory['states']:
            total += self.forward(state)
        return total

# Training loss (Bradley-Terry ranking model):
def ranking_loss(reward_net, traj_better, traj_worse):
    """
    P(traj_better > traj_worse) = exp(r_better) / (exp(r_better) + exp(r_worse))
    Loss = -log P(traj_better > traj_worse)
    """
    r_better = reward_net.predict_return(traj_better)
    r_worse = reward_net.predict_return(traj_worse)

    # Numerically stable version
    return torch.log(1 + torch.exp(r_worse - r_better))
```

### Step 4: Train Policy with Learned Reward

Use the learned reward function to train a new policy:

```python
# Instead of environment reward r_env, use learned reward r_learned
for episode in training_loop:
    state = env.reset()
    for step in range(max_steps):
        action = policy.act(state)
        next_state, r_env, done, info = env.step(action)

        # USE LEARNED REWARD instead of r_env
        r_learned = reward_net(state)

        # Standard RL update with learned reward
        policy.update(state, action, r_learned, next_state, done)

        state = next_state
        if done:
            break
```

---

## Implementation Architecture

### File Structure

```
RLagentLudo/
├── src/rl_agent_ludo/
│   ├── agents/
│   │   └── trex_agent.py              # NEW: T-REX policy agent
│   ├── preference_learning/           # NEW MODULE
│   │   ├── __init__.py
│   │   ├── reward_network.py          # Reward function learner
│   │   ├── trajectory_collector.py    # Collect & save trajectories
│   │   ├── trajectory_ranker.py       # Rank trajectories
│   │   └── trex_trainer.py            # Full T-REX training pipeline
│   └── environment/
│       └── level6_trex.py             # NEW: Level 6 environment
├── experiments/
│   ├── level6_collect_trajectories.py # Phase 1: Collect demos
│   ├── level6_learn_reward.py         # Phase 2-3: Learn reward
│   ├── level6_train_policy.py         # Phase 4: Train with learned reward
│   └── level6_evaluate.py             # Phase 5: Evaluate
├── checkpoints/
│   └── level6/
│       ├── trajectories/              # Saved trajectories
│       ├── reward_network.pth         # Learned reward function
│       └── trex_policy.pth            # Trained policy
└── docs/
    └── LEVEL6_TREX_IMPLEMENTATION_PLAN.md  # This file
```

---

## Phase-by-Phase Roadmap

## Phase 1: Trajectory Collection (Week 1, Days 1-3)

### Goal
Collect 1000+ trajectories from existing Level 1-5 agents playing against random opponents.

### Implementation

**File: `src/rl_agent_ludo/preference_learning/trajectory_collector.py`**

```python
"""
Trajectory Collector for T-REX.

Runs existing agents and records full game trajectories with metadata.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
import torch

class TrajectoryCollector:
    """Collects and saves game trajectories."""

    def __init__(self, save_dir: str = "checkpoints/level6/trajectories"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = []

    def collect_trajectory(self, env, agent, episode_id: int) -> Dict[str, Any]:
        """
        Run one episode and record full trajectory.

        Returns:
            trajectory: {
                'episode_id': int,
                'states': List[np.ndarray],
                'actions': List[int],
                'env_rewards': List[float],  # Ignore these for T-REX
                'outcome': 'win' or 'loss',
                'final_score': float,
                'num_captures': int,
                'episode_length': int,
                'agent_type': 'level5' or 'level3', etc.
            }
        """
        states = []
        actions = []
        env_rewards = []

        state, info = env.reset()
        done = False
        step = 0
        num_captures = info.get('captures', 0)

        while not done and step < 1000:
            # Record state
            states.append(state.copy())

            # Agent acts
            action = agent.act(state, greedy=False)
            actions.append(action)

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            env_rewards.append(reward)

            # Track captures
            if 'captured' in info and info['captured']:
                num_captures += 1

            state = next_state
            done = terminated or truncated
            step += 1

        # Determine outcome
        winner = info.get('winner', -1)
        outcome = 'win' if winner == 0 else 'loss'

        trajectory = {
            'episode_id': episode_id,
            'states': states,
            'actions': actions,
            'env_rewards': env_rewards,
            'outcome': outcome,
            'final_score': info.get('final_score', 0),
            'num_captures': num_captures,
            'episode_length': step,
            'agent_type': getattr(agent, 'name', 'unknown'),
        }

        return trajectory

    def collect_batch(self, env, agent, num_episodes: int,
                     batch_name: str = "default"):
        """Collect multiple trajectories and save."""
        print(f"Collecting {num_episodes} trajectories from {agent.name}...")

        batch_trajectories = []
        for i in range(num_episodes):
            traj = self.collect_trajectory(env, agent, episode_id=i)
            batch_trajectories.append(traj)

            if (i + 1) % 100 == 0:
                print(f"  Collected {i+1}/{num_episodes} trajectories")

        # Save batch
        save_path = self.save_dir / f"{batch_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(batch_trajectories, f)

        print(f"✅ Saved {len(batch_trajectories)} trajectories to {save_path}")

        self.trajectories.extend(batch_trajectories)
        return batch_trajectories

    def load_trajectories(self, batch_name: str) -> List[Dict]:
        """Load saved trajectories."""
        load_path = self.save_dir / f"{batch_name}.pkl"
        with open(load_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"✅ Loaded {len(trajectories)} trajectories from {load_path}")
        return trajectories
```

**File: `experiments/level6_collect_trajectories.py`**

```python
"""
Phase 1: Collect Trajectories from Existing Agents.

Run this script to generate training data for T-REX reward learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent
from rl_agent_ludo.preference_learning.trajectory_collector import TrajectoryCollector
import torch

def main():
    print("="*80)
    print("LEVEL 6 - PHASE 1: TRAJECTORY COLLECTION")
    print("="*80)

    # Environment (4-player full game)
    env = Level5MultiAgentLudo(num_players=4, tokens_per_player=2)

    # Collector
    collector = TrajectoryCollector(save_dir="checkpoints/level6/trajectories")

    # ========================
    # Collect from Level 5 Agent
    # ========================
    print("\n[1/3] Loading Level 5 trained agent...")
    level5_agent = SimpleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    level5_agent.load("checkpoints/level5/best_model.pth")
    level5_agent.name = "level5_trained"

    print("\n[1/3] Collecting 500 trajectories from Level 5 agent...")
    collector.collect_batch(env, level5_agent, num_episodes=500,
                           batch_name="level5_demos")

    # ========================
    # Collect from Random Agent (Negative Examples)
    # ========================
    print("\n[2/3] Collecting 300 trajectories from Random agent...")
    random_agent = RandomAgent(action_space=env.action_space)
    random_agent.name = "random"

    collector.collect_batch(env, random_agent, num_episodes=300,
                           batch_name="random_demos")

    # ========================
    # Collect from Level 3 Agent (Medium Skill)
    # ========================
    print("\n[3/3] Loading Level 3 agent for diversity...")
    from rl_agent_ludo.environment.level3_multitoken import Level3MultiTokenLudo

    env3 = Level3MultiTokenLudo(num_players=2, tokens_per_player=2)
    level3_agent = SimpleDQNAgent(
        state_dim=env3.observation_space.shape[0],
        action_dim=env3.action_space.n,
    )
    level3_agent.load("checkpoints/level3/best_model.pth")
    level3_agent.name = "level3_trained"

    # Adapt to Level 5 environment
    # (May need state/action space adaptation)
    # For now, collect from Level 3 environment
    collector_l3 = TrajectoryCollector(save_dir="checkpoints/level6/trajectories")
    collector_l3.collect_batch(env3, level3_agent, num_episodes=200,
                               batch_name="level3_demos")

    # ========================
    # Summary
    # ========================
    print("\n" + "="*80)
    print("TRAJECTORY COLLECTION COMPLETE!")
    print("="*80)
    print(f"✅ Level 5 demos: 500 trajectories")
    print(f"✅ Random demos: 300 trajectories")
    print(f"✅ Level 3 demos: 200 trajectories")
    print(f"📊 Total: 1000 trajectories")
    print(f"💾 Saved to: checkpoints/level6/trajectories/")
    print("\n➡️  Next step: Run level6_learn_reward.py")

if __name__ == "__main__":
    main()
```

### Deliverables (Days 1-3)
- ✅ `trajectory_collector.py` implemented
- ✅ `level6_collect_trajectories.py` implemented
- ✅ 1000+ trajectories collected and saved
- ✅ Trajectory statistics printed (win rates, avg length, etc.)

---

## Phase 2: Trajectory Ranking (Week 1, Days 4-5)

### Goal
Create preference pairs by ranking trajectories.

### Ranking Criteria

```python
def rank_trajectories(traj_i, traj_j) -> int:
    """
    Returns:
        1 if traj_i > traj_j (i is better)
        -1 if traj_j > traj_i (j is better)
        0 if equal (skip this pair)
    """
    # Rule 1: Win > Loss
    if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'loss':
        return 1
    if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'win':
        return -1

    # Rule 2: Among wins, more captures > fewer
    if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'win':
        if traj_i['num_captures'] > traj_j['num_captures']:
            return 1
        elif traj_i['num_captures'] < traj_j['num_captures']:
            return -1
        else:
            # Rule 3: Among equal captures, shorter > longer
            if traj_i['episode_length'] < traj_j['episode_length']:
                return 1
            elif traj_i['episode_length'] > traj_j['episode_length']:
                return -1

    # Rule 4: Among losses, survived longer > died early
    if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'loss':
        if traj_i['episode_length'] > traj_j['episode_length']:
            return 1
        elif traj_i['episode_length'] < traj_j['episode_length']:
            return -1

    return 0  # Equal
```

### Implementation

**File: `src/rl_agent_ludo/preference_learning/trajectory_ranker.py`**

```python
"""
Trajectory Ranker for T-REX.

Creates preference pairs from collected trajectories.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import random
import numpy as np

class TrajectoryRanker:
    """Ranks trajectories and creates preference pairs."""

    def __init__(self):
        self.preference_pairs = []

    def rank_pair(self, traj_i: Dict, traj_j: Dict) -> int:
        """
        Compare two trajectories.

        Returns:
            1 if traj_i > traj_j
            -1 if traj_j > traj_i
            0 if equal (skip)
        """
        # Rule 1: Win > Loss
        if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'loss':
            return 1
        if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'win':
            return -1

        # Rule 2: Among wins, more captures > fewer
        if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'win':
            if traj_i['num_captures'] > traj_j['num_captures']:
                return 1
            elif traj_i['num_captures'] < traj_j['num_captures']:
                return -1
            else:
                # Rule 3: Shorter episode > longer
                if traj_i['episode_length'] < traj_j['episode_length']:
                    return 1
                elif traj_i['episode_length'] > traj_j['episode_length']:
                    return -1

        # Rule 4: Among losses, survived longer > died early
        if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'loss':
            if traj_i['episode_length'] > traj_j['episode_length']:
                return 1
            elif traj_i['episode_length'] < traj_j['episode_length']:
                return -1

        return 0  # Equal

    def create_preference_pairs(self, trajectories: List[Dict],
                               max_pairs: int = 10000) -> List[Tuple]:
        """
        Create preference pairs from trajectories.

        Args:
            trajectories: List of trajectory dicts
            max_pairs: Maximum number of pairs to create

        Returns:
            preference_pairs: List of (better_traj, worse_traj) tuples
        """
        print(f"Creating preference pairs from {len(trajectories)} trajectories...")

        preference_pairs = []

        # Sample pairs randomly
        for _ in range(max_pairs):
            i, j = random.sample(range(len(trajectories)), 2)
            traj_i = trajectories[i]
            traj_j = trajectories[j]

            ranking = self.rank_pair(traj_i, traj_j)

            if ranking == 1:
                # i > j
                preference_pairs.append((traj_i, traj_j))
            elif ranking == -1:
                # j > i
                preference_pairs.append((traj_j, traj_i))
            # If ranking == 0, skip this pair

        print(f"✅ Created {len(preference_pairs)} preference pairs")

        # Statistics
        self._print_statistics(preference_pairs)

        self.preference_pairs = preference_pairs
        return preference_pairs

    def _print_statistics(self, pairs: List[Tuple]):
        """Print statistics about preference pairs."""
        print("\n📊 Preference Pair Statistics:")

        better_wins = sum(1 for better, worse in pairs if better['outcome'] == 'win')
        worse_losses = sum(1 for better, worse in pairs if worse['outcome'] == 'loss')

        print(f"  Total pairs: {len(pairs)}")
        print(f"  Better trajectories that won: {better_wins} ({better_wins/len(pairs)*100:.1f}%)")
        print(f"  Worse trajectories that lost: {worse_losses} ({worse_losses/len(pairs)*100:.1f}%)")

        # Average metrics
        avg_better_captures = np.mean([b['num_captures'] for b, w in pairs])
        avg_worse_captures = np.mean([w['num_captures'] for b, w in pairs])

        print(f"  Avg captures (better): {avg_better_captures:.2f}")
        print(f"  Avg captures (worse): {avg_worse_captures:.2f}")

    def save_pairs(self, filepath: str):
        """Save preference pairs to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.preference_pairs, f)
        print(f"✅ Saved {len(self.preference_pairs)} pairs to {filepath}")

    def load_pairs(self, filepath: str):
        """Load preference pairs from disk."""
        with open(filepath, 'rb') as f:
            self.preference_pairs = pickle.load(f)
        print(f"✅ Loaded {len(self.preference_pairs)} pairs from {filepath}")
        return self.preference_pairs
```

### Deliverables (Days 4-5)
- ✅ `trajectory_ranker.py` implemented
- ✅ 10,000 preference pairs created
- ✅ Ranking statistics validated
- ✅ Preference pairs saved

---

## Phase 3: Learn Reward Network (Week 2, Days 1-5)

### Goal
Train a neural network to predict trajectory returns consistent with preference rankings.

### Implementation

**File: `src/rl_agent_ludo/preference_learning/reward_network.py`**

```python
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

class RewardNetwork(nn.Module):
    """Neural network that predicts scalar reward for each state."""

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Scalar reward
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
    """Trains reward network from preference pairs."""

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128,
                 learning_rate: float = 3e-4, device: str = 'cpu'):
        self.device = device
        self.reward_net = RewardNetwork(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def ranking_loss(self, traj_better: Dict, traj_worse: Dict) -> torch.Tensor:
        """
        Bradley-Terry ranking loss.

        Loss = -log P(better > worse)
             = -log(exp(r_better) / (exp(r_better) + exp(r_worse)))
             = log(1 + exp(r_worse - r_better))

        Args:
            traj_better: Better trajectory dict
            traj_worse: Worse trajectory dict

        Returns:
            loss: scalar
        """
        # Convert states to tensors
        states_better = torch.FloatTensor(traj_better['states']).to(self.device)
        states_worse = torch.FloatTensor(traj_worse['states']).to(self.device)

        # Predict returns
        r_better = self.reward_net.predict_return(states_better)
        r_worse = self.reward_net.predict_return(states_worse)

        # Ranking loss (numerically stable)
        loss = torch.log(1 + torch.exp(r_worse - r_better))

        return loss

    def train_epoch(self, preference_pairs: List[Tuple], batch_size: int = 32):
        """Train for one epoch."""
        self.reward_net.train()

        # Shuffle pairs
        import random
        random.shuffle(preference_pairs)

        epoch_losses = []

        # Mini-batch training
        for i in range(0, len(preference_pairs), batch_size):
            batch = preference_pairs[i:i+batch_size]

            batch_loss = 0
            for traj_better, traj_worse in batch:
                loss = self.ranking_loss(traj_better, traj_worse)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)

            # Backprop
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(batch_loss.item())

        return np.mean(epoch_losses)

    def validate(self, preference_pairs: List[Tuple]) -> float:
        """Compute validation loss."""
        self.reward_net.eval()

        val_losses = []
        with torch.no_grad():
            for traj_better, traj_worse in preference_pairs:
                loss = self.ranking_loss(traj_better, traj_worse)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def train(self, train_pairs: List[Tuple], val_pairs: List[Tuple],
             num_epochs: int = 100, batch_size: int = 32):
        """
        Full training loop.

        Args:
            train_pairs: Training preference pairs
            val_pairs: Validation preference pairs
            num_epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"Training reward network for {num_epochs} epochs...")
        print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_pairs, batch_size)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_pairs)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.reward_net.state_dict(),
                          'checkpoints/level6/reward_network_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.reward_net.load_state_dict(
            torch.load('checkpoints/level6/reward_network_best.pth')
        )
        print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")

    def save(self, filepath: str):
        """Save reward network."""
        torch.save({
            'state_dict': self.reward_net.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
        print(f"✅ Saved reward network to {filepath}")

    def load(self, filepath: str):
        """Load reward network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.reward_net.load_state_dict(checkpoint['state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"✅ Loaded reward network from {filepath}")
```

**File: `experiments/level6_learn_reward.py`**

```python
"""
Phase 2-3: Rank Trajectories and Learn Reward Function.

This script:
1. Loads collected trajectories
2. Creates preference pairs
3. Trains reward network
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.preference_learning.trajectory_collector import TrajectoryCollector
from rl_agent_ludo.preference_learning.trajectory_ranker import TrajectoryRanker
from rl_agent_ludo.preference_learning.reward_network import RewardLearner
import torch

def main():
    print("="*80)
    print("LEVEL 6 - PHASE 2-3: LEARN REWARD FUNCTION")
    print("="*80)

    # ========================
    # Load Trajectories
    # ========================
    print("\n[1/3] Loading trajectories...")
    collector = TrajectoryCollector()

    trajs_l5 = collector.load_trajectories("level5_demos")
    trajs_random = collector.load_trajectories("random_demos")
    trajs_l3 = collector.load_trajectories("level3_demos")

    all_trajectories = trajs_l5 + trajs_random + trajs_l3
    print(f"✅ Loaded {len(all_trajectories)} total trajectories")

    # ========================
    # Create Preference Pairs
    # ========================
    print("\n[2/3] Creating preference pairs...")
    ranker = TrajectoryRanker()
    preference_pairs = ranker.create_preference_pairs(
        all_trajectories,
        max_pairs=10000
    )

    # Split train/val
    split_idx = int(0.8 * len(preference_pairs))
    train_pairs = preference_pairs[:split_idx]
    val_pairs = preference_pairs[split_idx:]

    print(f"✅ Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Save pairs
    ranker.save_pairs("checkpoints/level6/preference_pairs.pkl")

    # ========================
    # Train Reward Network
    # ========================
    print("\n[3/3] Training reward network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    learner = RewardLearner(
        state_dim=16,  # Ludo state dimension
        hidden_dim=128,
        learning_rate=3e-4,
        device=device
    )

    learner.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_epochs=100,
        batch_size=32
    )

    # Save final model
    learner.save("checkpoints/level6/reward_network_final.pth")

    # ========================
    # Summary
    # ========================
    print("\n" + "="*80)
    print("REWARD LEARNING COMPLETE!")
    print("="*80)
    print(f"✅ Trained on {len(train_pairs)} preference pairs")
    print(f"✅ Final validation loss: {learner.val_losses[-1]:.4f}")
    print(f"💾 Saved to: checkpoints/level6/reward_network_final.pth")
    print("\n➡️  Next step: Run level6_train_policy.py")

if __name__ == "__main__":
    main()
```

### Deliverables (Week 2)
- ✅ `reward_network.py` implemented
- ✅ `level6_learn_reward.py` script working
- ✅ Reward network trained and saved
- ✅ Training curves plotted
- ✅ Validation loss converged

---

## Phase 4: Train Policy with Learned Reward (Week 3)

### Goal
Train a DQN agent using the learned reward function instead of environment rewards.

### Implementation

**File: `src/rl_agent_ludo/agents/trex_agent.py`**

```python
"""
T-REX Agent: DQN trained with learned reward function.

Identical to SimpleDQNAgent, but uses learned reward instead of env reward.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent, DQNNetwork
from rl_agent_ludo.preference_learning.reward_network import RewardNetwork

class TREXAgent(SimpleDQNAgent):
    """
    DQN agent that uses learned reward function from T-REX.

    Inherits from SimpleDQNAgent but replaces environment reward
    with reward from learned reward network.
    """

    def __init__(self, reward_network_path: str, *args, **kwargs):
        """
        Initialize T-REX agent.

        Args:
            reward_network_path: Path to trained reward network
            *args, **kwargs: Same as SimpleDQNAgent
        """
        super().__init__(*args, **kwargs)

        # Load learned reward network
        self.reward_net = RewardNetwork(
            state_dim=self.state_dim,
            hidden_dim=128
        ).to(self.device)

        checkpoint = torch.load(reward_network_path, map_location=self.device)
        self.reward_net.load_state_dict(checkpoint['state_dict'])
        self.reward_net.eval()  # Freeze reward network

        print(f"✅ Loaded learned reward function from {reward_network_path}")

    def get_learned_reward(self, state: np.ndarray) -> float:
        """
        Get reward from learned reward network.

        Args:
            state: Current state

        Returns:
            learned_reward: Scalar reward from network
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            reward = self.reward_net(state_tensor)
            return reward.item()

    def store_transition(self, state: np.ndarray, action: int,
                        env_reward: float, next_state: np.ndarray, done: bool):
        """
        Store transition using LEARNED REWARD instead of env_reward.

        Args:
            state: Current state
            action: Action taken
            env_reward: Environment reward (IGNORED!)
            next_state: Next state
            done: Episode done flag
        """
        # Replace env_reward with learned reward
        learned_reward = self.get_learned_reward(state)

        # Store with learned reward
        super().store_transition(state, action, learned_reward, next_state, done)
```

**File: `experiments/level6_train_policy.py`**

```python
"""
Phase 4: Train Policy with Learned Reward.

Train a DQN agent using the T-REX learned reward function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.trex_agent import TREXAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent
import torch
import numpy as np

def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate agent performance."""
    wins = 0
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state
            done = terminated or truncated

        if info.get('winner') == 0:
            wins += 1
        total_rewards.append(episode_reward)

    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)

    return win_rate, avg_reward

def main():
    print("="*80)
    print("LEVEL 6 - PHASE 4: TRAIN POLICY WITH LEARNED REWARD")
    print("="*80)

    # Environment
    env = Level5MultiAgentLudo(num_players=4, tokens_per_player=2)

    # T-REX Agent (uses learned reward)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = TREXAgent(
        reward_network_path="checkpoints/level6/reward_network_final.pth",
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=5e-5,
        epsilon=1.0,
        epsilon_decay=0.995,
        device=device
    )

    # Training
    num_episodes = 15000
    eval_freq = 1000

    print(f"\nTraining for {num_episodes} episodes...")
    print(f"Evaluation frequency: {eval_freq} episodes\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Act
            action = agent.act(state)
            next_state, env_reward, terminated, truncated, info = env.step(action)

            # Store with LEARNED reward (done inside TREXAgent)
            agent.store_transition(state, action, env_reward, next_state, terminated or truncated)

            # Learn
            loss = agent.learn()

            episode_reward += env_reward
            state = next_state
            done = terminated or truncated

        # Decay epsilon
        agent.decay_epsilon()

        # Logging
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {agent.get_avg_loss():.4f}")

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION at Episode {episode+1}")
            print(f"{'='*60}")

            win_rate, avg_reward = evaluate_agent(env, agent, num_episodes=200)

            print(f"Win Rate: {win_rate*100:.1f}%")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"{'='*60}\n")

            # Save checkpoint
            agent.save(f"checkpoints/level6/trex_policy_ep{episode+1}.pth")

    # Final save
    agent.save("checkpoints/level6/trex_policy_final.pth")

    print("\n" + "="*80)
    print("POLICY TRAINING COMPLETE!")
    print("="*80)
    print(f"💾 Saved to: checkpoints/level6/trex_policy_final.pth")
    print("\n➡️  Next step: Run level6_evaluate.py")

if __name__ == "__main__":
    main()
```

### Deliverables (Week 3)
- ✅ `trex_agent.py` implemented
- ✅ `level6_train_policy.py` working
- ✅ Policy trained for 15k episodes
- ✅ Training curves saved
- ✅ Checkpoints saved

---

## Phase 5: Evaluation (Week 4)

### Goal
Compare T-REX agent against Level 5 baseline.

**File: `experiments/level6_evaluate.py`**

```python
"""
Phase 5: Comprehensive Evaluation of T-REX Agent.

Compare:
- T-REX agent (Level 6)
- Level 5 baseline
- Random agent

Metrics:
- Win rate
- Captures
- Episode length
- Head-to-head matchup
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.trex_agent import TREXAgent
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent
import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_comprehensive(env, agent, num_episodes=400):
    """Comprehensive evaluation."""
    results = {
        'wins': 0,
        'losses': 0,
        'captures': [],
        'episode_lengths': [],
        'final_scores': []
    }

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        captures = 0
        steps = 0

        while not done:
            action = agent.act(state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            if 'captured' in info and info['captured']:
                captures += 1

            state = next_state
            done = terminated or truncated
            steps += 1

        # Record
        if info.get('winner') == 0:
            results['wins'] += 1
        else:
            results['losses'] += 1

        results['captures'].append(captures)
        results['episode_lengths'].append(steps)
        results['final_scores'].append(info.get('final_score', 0))

    # Compute statistics
    stats = {
        'win_rate': results['wins'] / num_episodes,
        'avg_captures': np.mean(results['captures']),
        'avg_episode_length': np.mean(results['episode_lengths']),
        'avg_final_score': np.mean(results['final_scores'])
    }

    return stats, results

def main():
    print("="*80)
    print("LEVEL 6 - PHASE 5: COMPREHENSIVE EVALUATION")
    print("="*80)

    env = Level5MultiAgentLudo(num_players=4, tokens_per_player=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========================
    # Load Agents
    # ========================
    print("\n[1/3] Loading agents...")

    # T-REX Agent (Level 6)
    trex_agent = TREXAgent(
        reward_network_path="checkpoints/level6/reward_network_final.pth",
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )
    trex_agent.load("checkpoints/level6/trex_policy_final.pth")
    print("✅ Loaded T-REX agent")

    # Level 5 Baseline
    level5_agent = SimpleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )
    level5_agent.load("checkpoints/level5/best_model.pth")
    print("✅ Loaded Level 5 baseline")

    # Random Baseline
    random_agent = RandomAgent(action_space=env.action_space)
    print("✅ Random baseline ready")

    # ========================
    # Evaluate All Agents
    # ========================
    print("\n[2/3] Evaluating agents (400 episodes each)...")

    print("\nEvaluating T-REX agent...")
    stats_trex, _ = evaluate_comprehensive(env, trex_agent, num_episodes=400)

    print("Evaluating Level 5 baseline...")
    stats_l5, _ = evaluate_comprehensive(env, level5_agent, num_episodes=400)

    print("Evaluating Random baseline...")
    stats_random, _ = evaluate_comprehensive(env, random_agent, num_episodes=400)

    # ========================
    # Print Results
    # ========================
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\n{'Agent':<20} {'Win Rate':<15} {'Avg Captures':<15} {'Avg Length'}")
    print("-" * 70)
    print(f"{'T-REX (Level 6)':<20} {stats_trex['win_rate']*100:>6.1f}%        "
          f"{stats_trex['avg_captures']:>6.2f}         {stats_trex['avg_episode_length']:>6.1f}")
    print(f"{'Level 5 Baseline':<20} {stats_l5['win_rate']*100:>6.1f}%        "
          f"{stats_l5['avg_captures']:>6.2f}         {stats_l5['avg_episode_length']:>6.1f}")
    print(f"{'Random':<20} {stats_random['win_rate']*100:>6.1f}%        "
          f"{stats_random['avg_captures']:>6.2f}         {stats_random['avg_episode_length']:>6.1f}")

    # Compute improvement
    improvement = (stats_trex['win_rate'] - stats_l5['win_rate']) * 100
    print(f"\n🎯 T-REX Improvement over Level 5: {improvement:+.1f}%")

    # ========================
    # Head-to-Head Comparison
    # ========================
    print("\n[3/3] Head-to-head: T-REX vs Level 5 (200 games)...")
    # TODO: Implement head-to-head matches

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
```

### Deliverables (Week 4)
- ✅ Comprehensive evaluation complete
- ✅ Comparison plots generated
- ✅ Results documented
- ✅ Paper-ready figures

---

## Expected Results

### Hypothesis

T-REX should achieve **63-67% win rate** (vs Level 5's 61%) because:
1. Learns from best behaviors of all demonstrators
2. Can extrapolate beyond demonstrations
3. More robust reward signal

### Success Criteria

✅ **Minimum**: Match Level 5 performance (61%)
🎯 **Target**: Beat Level 5 by 2-5% (63-66%)
🏆 **Stretch**: Beat Level 5 by 5%+ (66%+)

### What to Show Your Professor

1. **Trajectory Collection**: 1000+ demos from existing agents
2. **Preference Learning**: Reward network trained on rankings
3. **Performance Improvement**: T-REX beats baseline
4. **Learned Reward Visualization**: What did the network learn?
5. **Ablation Studies**: Effect of demo quality, number of pairs, etc.

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 (Days 1-3) | Collect Trajectories | 1000+ trajectories saved |
| 1 (Days 4-5) | Rank Trajectories | 10k preference pairs |
| 2 | Learn Reward | Trained reward network |
| 3 | Train Policy | T-REX agent trained |
| 4 | Evaluate | Results & plots ready |

**Total**: 3-4 weeks to complete implementation + evaluation

---

## Next Steps

1. **Read T-REX Paper**: https://arxiv.org/abs/1904.06387
2. **Browse Original Code**: https://github.com/dsbrown1331/trex
3. **Start Implementation**: Begin with Phase 1 (trajectory collection)

Ready to start implementing? Let me know and I'll help you build it step by step! 🚀
