# Level 6: T-REX Quick Start Guide

**Fast-track guide to get T-REX running ASAP.**

For full details, see [LEVEL6_TREX_IMPLEMENTATION_PLAN.md](LEVEL6_TREX_IMPLEMENTATION_PLAN.md)

---

## What is T-REX?

**T-REX** = Learn reward from ranked trajectories → Train policy with learned reward

**Input**: Game trajectories ranked by quality (win > loss)
**Output**: Agent that exceeds demonstrator performance

**Timeline**: 3-4 weeks full implementation

---

## 5-Step Implementation

### Step 1: Create Directory Structure (5 minutes)

```bash
# Create directories
mkdir -p src/rl_agent_ludo/preference_learning
mkdir -p checkpoints/level6/trajectories
mkdir -p experiments/level6

# Create __init__.py
touch src/rl_agent_ludo/preference_learning/__init__.py
```

### Step 2: Collect Trajectories (2-3 hours)

**Files to create:**
1. `src/rl_agent_ludo/preference_learning/trajectory_collector.py`
2. `experiments/level6_collect_trajectories.py`

**Run:**
```bash
python experiments/level6_collect_trajectories.py
```

**Expected output:**
- 1000+ trajectories saved to `checkpoints/level6/trajectories/`
- Mix of Level 5, Level 3, and random agent demos

### Step 3: Learn Reward Function (1 week)

**Files to create:**
1. `src/rl_agent_ludo/preference_learning/trajectory_ranker.py`
2. `src/rl_agent_ludo/preference_learning/reward_network.py`
3. `experiments/level6_learn_reward.py`

**Run:**
```bash
python experiments/level6_learn_reward.py
```

**Expected output:**
- Trained reward network saved to `checkpoints/level6/reward_network_final.pth`
- Training curves showing convergence

### Step 4: Train Policy (1-2 weeks)

**Files to create:**
1. `src/rl_agent_ludo/agents/trex_agent.py`
2. `experiments/level6_train_policy.py`

**Run:**
```bash
python experiments/level6_train_policy.py
```

**Expected output:**
- Trained T-REX agent saved to `checkpoints/level6/trex_policy_final.pth`
- Win rate improving during training

### Step 5: Evaluate (3-4 days)

**Files to create:**
1. `experiments/level6_evaluate.py`

**Run:**
```bash
python experiments/level6_evaluate.py
```

**Expected results:**
- T-REX: **63-67% win rate** 🎯
- Level 5: 61% win rate
- Improvement: +2-6%

---

## Key Implementation Details

### Trajectory Format

```python
trajectory = {
    'states': [s0, s1, ..., sT],        # Full state sequence
    'actions': [a0, a1, ..., aT],       # Action sequence
    'outcome': 'win' or 'loss',         # Game result
    'num_captures': 5,                  # Captures made
    'episode_length': 120               # Steps
}
```

### Ranking Logic

```python
def rank(traj_i, traj_j):
    # 1. Win > Loss
    if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'loss':
        return 1  # i is better

    # 2. Among wins: more captures > fewer
    if both_won and traj_i['num_captures'] > traj_j['num_captures']:
        return 1

    # 3. Among wins: shorter > longer
    if equal_captures and traj_i['episode_length'] < traj_j['episode_length']:
        return 1

    return -1  # j is better
```

### Reward Network

```python
class RewardNetwork(nn.Module):
    def __init__(self, state_dim=16):
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Scalar reward
        )

    def predict_return(self, trajectory):
        # Sum rewards over trajectory
        return sum(self.network(state) for state in trajectory['states'])
```

### Training Loss

```python
def ranking_loss(reward_net, traj_better, traj_worse):
    """Bradley-Terry ranking loss"""
    r_better = reward_net.predict_return(traj_better)
    r_worse = reward_net.predict_return(traj_worse)

    # Loss = -log P(better > worse)
    loss = torch.log(1 + torch.exp(r_worse - r_better))
    return loss
```

---

## Critical Implementation Notes

### ⚠️ Common Pitfalls

1. **Don't use environment rewards during policy training**
   - Use learned reward instead: `r = reward_net(state)`

2. **Freeze reward network during policy training**
   - `reward_net.eval()` and `no_grad()`

3. **Collect diverse demonstrations**
   - Mix of good (Level 5), medium (Level 3), and bad (random) trajectories

4. **Sufficient preference pairs**
   - Need 5k-10k pairs for good learning

5. **State normalization**
   - Ensure states are normalized the same way across collection and training

### 🎯 Success Metrics

**Phase 1 (Trajectory Collection):**
- ✅ 1000+ trajectories collected
- ✅ Win rate distribution: ~60% Level 5, ~25% random

**Phase 2 (Reward Learning):**
- ✅ Validation loss converges
- ✅ Ranking accuracy > 80%

**Phase 3 (Policy Training):**
- ✅ Training stable (loss decreases)
- ✅ Win rate improves over episodes

**Phase 4 (Evaluation):**
- ✅ T-REX beats Level 5 baseline
- ✅ Statistical significance (p < 0.05)

---

## Code Templates

### Minimal Trajectory Collector

```python
class TrajectoryCollector:
    def collect_trajectory(self, env, agent):
        states, actions = [], []
        state, _ = env.reset()
        done = False

        while not done:
            states.append(state.copy())
            action = agent.act(state)
            actions.append(action)
            state, _, done, _, info = env.step(action)

        return {
            'states': states,
            'actions': actions,
            'outcome': 'win' if info['winner'] == 0 else 'loss'
        }
```

### Minimal Reward Network Training

```python
# Create preference pairs
pairs = [(traj_better, traj_worse), ...]

# Train
reward_net = RewardNetwork(state_dim=16)
optimizer = Adam(reward_net.parameters(), lr=3e-4)

for epoch in range(100):
    for traj_b, traj_w in pairs:
        loss = ranking_loss(reward_net, traj_b, traj_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Minimal T-REX Agent

```python
class TREXAgent(SimpleDQNAgent):
    def __init__(self, reward_net_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_net = RewardNetwork()
        self.reward_net.load(reward_net_path)
        self.reward_net.eval()

    def store_transition(self, state, action, env_reward, next_state, done):
        # Use learned reward instead of env_reward
        learned_reward = self.reward_net(state).item()
        super().store_transition(state, action, learned_reward, next_state, done)
```

---

## Timeline Checklist

### Week 1
- [ ] Day 1-2: Implement trajectory collector
- [ ] Day 3: Collect 1000+ trajectories
- [ ] Day 4-5: Implement trajectory ranker and create preference pairs

### Week 2
- [ ] Day 1-2: Implement reward network
- [ ] Day 3-5: Train reward network to convergence

### Week 3
- [ ] Day 1-2: Implement T-REX agent
- [ ] Day 3-7: Train policy for 15k episodes

### Week 4
- [ ] Day 1-2: Comprehensive evaluation
- [ ] Day 3-4: Generate plots and results
- [ ] Day 5-7: Document findings for professor

---

## Expected Conversation with Professor

**You**: "I implemented T-REX for Level 6, which learns a reward function from ranked trajectories."

**Prof**: "How does it compare to your Level 5 baseline?"

**You**: "T-REX achieved 65% win rate vs Level 5's 61%, a 4% improvement. It learned to prioritize aggressive captures early and defensive play late."

**Prof**: "What if the demonstrations are suboptimal?"

**You**: "That's the beauty of T-REX - it extrapolates beyond demonstrations. Even though some demos were from random agents, the learned reward still improved performance."

**Prof**: "Can you show the learned reward?"

**You**: "Yes, I visualized the reward network. It assigns high rewards to states with tokens near the goal and low rewards to risky positions."

---

## Troubleshooting

### Issue: Reward network not converging
**Solution**:
- Increase training epochs (100 → 200)
- Check preference pair quality (should have clear rankings)
- Add more preference pairs

### Issue: Policy not improving
**Solution**:
- Verify learned rewards are actually being used (not env rewards)
- Check reward magnitude (should be in reasonable range)
- Ensure reward network is frozen during policy training

### Issue: T-REX worse than baseline
**Solution**:
- Check demo quality (need good trajectories)
- Increase preference pair diversity
- Try different ranking criteria

---

## Resources

- **Full Plan**: [LEVEL6_TREX_IMPLEMENTATION_PLAN.md](LEVEL6_TREX_IMPLEMENTATION_PLAN.md)
- **T-REX Paper**: https://arxiv.org/abs/1904.06387
- **Original Code**: https://github.com/dsbrown1331/trex
- **Your Research**: `.projectDescription/Research/`

---

Ready to start? Begin with Step 1! 🚀
