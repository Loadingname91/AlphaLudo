# Comparison: Current Implementation vs Reference Repository

## Performance Comparison

### Reference Repository Results (2p2t)
- **Q-Agent vs Random**: 64.58% win rate
- **Q-Self-Agent vs Random**: 58.8% win rate  
- **Q-Self-Agent vs Q-Agent**: 45.6% win rate
- **Hyperparameters**: ε=0.1, α=0.3, γ=0.5

### Current Implementation Results (2p2t)
- **Phase 1 Eval vs Random**: 50.24% win rate
- **Phase 2 Eval vs Random**: 52.1% win rate
- **Phase 2 Eval vs Phase 1**: 52.06% win rate
- **Hyperparameters**: ε=0.0938, α=0.1418, γ=0.7349

**Performance Gap**: ~12-14% lower win rate compared to reference

---

## Key Differences Identified

### 1. Reward Structure (CRITICAL DIFFERENCE)

#### Reference Repository Reward System
The reference uses a **rich, explicit reward structure** with large reward values:

| Action | Reward | Description |
|--------|--------|-------------|
| Successful Defense | +5 | Defending token under attack |
| Failed Defense | -5 | Token captured despite defense attempt |
| Capture Enemy | +7 | Capturing enemy token |
| Being Captured | -7 | Agent's token captured |
| Exit HOME | +18 | Moving token out of base |
| Reach SAFE Zone | +5 | Moving token to safe position |
| Reach GOAL | +20 | Moving token to goal |
| Favorable Move | +30 | Moving furthest token when indecisive |
| Unfavorable Move | -30 | Poor strategic choice |
| Win Game | +50 | Winning the game |

**Total reward range**: -30 to +50+ (can accumulate multiple rewards per turn)

#### Current Implementation Reward System
The current implementation uses **sparse environment rewards** with internal scaling:

- **Environment**: +1 (win), -1 (loss), 0 (ongoing)
- **Internal Scaling**: Base reward × 10.0 + tactical bonuses
- **Tactical Bonuses**:
  - POT_KILL: +0.5 (max +1.0 with context)
  - POT_SAFETY: +0.3
  - POT_BOOST: +0.2
  - POT_GOAL: +1.0
  - POT_RISK: -0.2 (max -0.6 with context)

**Total reward range**: ~-0.6 to +11 (much smaller magnitude)

#### Impact Analysis
1. **Signal Strength**: Reference rewards are 5-50x larger, providing stronger learning signals
2. **Reward Density**: Reference provides immediate feedback for every strategic action
3. **Learning Speed**: Larger rewards → faster Q-value updates → faster convergence
4. **Strategic Guidance**: Reference explicitly rewards defense/offense strategy

### 2. State Space Representation

#### Reference Repository (Zone-Based)
- **Observation Space**: 8-tuple `(HOME, PATH, SAFE, GOAL, EV1, EV2, TV1, TV2)`
  - HOME, PATH, SAFE, GOAL: Counts (0-2 for 2 tokens)
  - EV1, EV2: Enemy vulnerable flags (0-1)
  - TV1, TV2: Token under attack flags (0-1)
- **Action Space**: 2-tuple `(MOVE_TOKEN1, MOVE_TOKEN2)` - boolean flags
- **State Space Size**: ~2^8 = 256 states (very small, fully explorable)

#### Current Implementation (Combined State)
- **Observation Space**: 13-tuple `(HOME, PATH, SAFE, GOAL, P1, P2, P3, P4, T1, T2, T3, T4, Context)`
  - Zone counts: 0-4 (but clamped for 2 tokens)
  - Potentials: 7 values per piece (0-6)
  - Threat flags: 0-1 per piece
  - Context: 0-2
- **Action Space**: Discrete(4) - piece index selection
- **State Space Size**: ~2.4M states (much larger, harder to explore)

#### Impact Analysis
1. **Exploration**: Reference's smaller state space is fully explorable in fewer episodes
2. **Generalization**: Current implementation may need more samples per state
3. **Simplicity**: Reference's simpler state may be more effective for tabular Q-learning

### 3. Game Rules Differences

#### Reference Repository
- **Simplified Rules**: 
  - No requirement to roll 6 to exit base
  - Single token can capture multiple enemies on same tile
  - Game ends when one player has 2 tokens in GOAL

#### Current Implementation
- **Standard Rules**: 
  - Requires rolling 6 to exit base (standard Ludo rules)
  - Standard capture rules
  - Game ends when all tokens reach GOAL

#### Impact Analysis
1. **Game Complexity**: Reference's simplified rules make learning easier
2. **State Transitions**: Fewer invalid states in reference (no "stuck in base" scenarios)
3. **Episode Length**: Reference games may be shorter, allowing more training episodes

### 4. Training Configuration

#### Reference Repository
- **Training Episodes**: Not explicitly stated, but likely 10k-50k
- **Epsilon Schedule**: Fixed ε=0.1 (no decay mentioned)
- **Evaluation**: 1000 episodes with greedy policy

#### Current Implementation
- **Training Episodes**: 50k (Phase 1) + 25k (Phase 2) = 75k total
- **Epsilon Schedule**: Linear decay from ε=0.0938 to ε=0.043
- **Evaluation**: 1000 episodes with greedy policy (ε=0.0)

#### Impact Analysis
1. **Exploration**: Current implementation uses epsilon decay, which may reduce exploration too early
2. **Training Duration**: Current implementation trains longer but may not converge as well

### 5. Hyperparameters

#### Reference Repository
- **Learning Rate (α)**: 0.3 (higher)
- **Discount Factor (γ)**: 0.5 (lower, more myopic)
- **Epsilon (ε)**: 0.1 (fixed)

#### Current Implementation
- **Learning Rate (α)**: 0.1418 (lower, from hyperparameter optimization)
- **Discount Factor (γ)**: 0.7349 (higher, more long-term thinking)
- **Epsilon (ε)**: 0.0938 → 0.043 (decaying)

#### Impact Analysis
1. **Learning Rate**: Reference's higher α allows faster Q-value updates
2. **Discount Factor**: Reference's lower γ focuses on immediate rewards (matches their reward structure)
3. **Epsilon**: Reference's fixed ε maintains exploration throughout training

---

## Root Cause Analysis

### Primary Issue: Reward Structure
The **most critical difference** is the reward structure:

1. **Magnitude**: Reference rewards are 5-50x larger, providing much stronger learning signals
2. **Density**: Reference provides immediate feedback for every strategic action (defense, offense, progress)
3. **Strategy Alignment**: Reference explicitly encodes the defense/offense strategy in rewards

### Secondary Issues:
1. **State Space Complexity**: Current implementation's larger state space requires more exploration
2. **Hyperparameters**: Current implementation's lower learning rate and higher discount factor may not match the sparse reward structure
3. **Game Rules**: Reference's simplified rules reduce complexity

---

## Recommendations for Improvement

### 1. Implement Rich Reward Structure (HIGH PRIORITY)
Modify the environment or agent to provide rewards matching the reference:

```python
# In environment or reward wrapper:
def compute_reward(state, action, next_state, done):
    reward = 0.0
    
    # Defense rewards
    if token_was_under_attack and successfully_defended:
        reward += 5
    elif token_was_under_attack and was_captured:
        reward -= 5
    
    # Offense rewards
    if captured_enemy:
        reward += 7
    if was_captured:
        reward -= 7
    
    # Progress rewards
    if exited_home:
        reward += 18
    if reached_safe_zone:
        reward += 5
    if reached_goal:
        reward += 20
    
    # Strategic rewards
    if favorable_move:
        reward += 30
    if unfavorable_move:
        reward -= 30
    
    # Win/loss
    if done:
        if won:
            reward += 50
        else:
            reward -= 50  # or keep -1
    
    return reward
```

### 2. Simplify State Space for 2p2t (MEDIUM PRIORITY)
For 2-token games, use the simpler zone-based state space:
- Match reference's 8-tuple representation
- Reduces state space from ~2.4M to ~256 states
- Enables full exploration in fewer episodes

### 3. Adjust Hyperparameters (MEDIUM PRIORITY)
For rich reward structure:
- Increase learning rate: α = 0.3 (match reference)
- Decrease discount factor: γ = 0.5 (match reference)
- Consider fixed epsilon: ε = 0.1 (no decay)

### 4. Consider Simplified Game Rules (LOW PRIORITY)
If matching reference exactly:
- Remove requirement to roll 6 to exit base
- Allow single token to capture multiple enemies
- End game when 2 tokens reach GOAL

---

## Expected Impact

Implementing the rich reward structure should:
- **Increase win rate by 10-15%** (from ~52% to ~62-67%)
- **Faster convergence** (fewer episodes needed)
- **Better strategic play** (explicit defense/offense guidance)

The combination of rich rewards + simpler state space + matching hyperparameters should bring performance close to the reference repository's 64.58% win rate.

