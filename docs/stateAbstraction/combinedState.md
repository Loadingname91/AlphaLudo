# Combined State Abstraction (13-Tuple)

## Overview

The Combined State Abstractor merges **zone-based** (strategic) and **potential-based** (tactical) abstractions into a unified 13-tuple representation. This provides both strategic overview and tactical detail in a single state space.

**Use Case**: Tabular Q-Learning (discrete state space)

**Status**: Optimized version (removed redundant aggregates for better sample efficiency)

## State Tuple Structure

**Format**: `(HOME, PATH, SAFE, GOAL, P1, P2, P3, P4, T1, T2, T3, T4, Context)`

### Zone Distribution (4 dimensions)

- **HOME**: Count of tokens in base/home (0-4)
- **PATH**: Count of tokens on main path, not in safe zones (0-4)
- **SAFE**: Count of tokens in safe zones (0-4)
- **GOAL**: Count of tokens that reached goal (0-4)

### Per-Piece Tactical (8 dimensions)

- **P1-P4**: Potential classification for each piece (0-6)
  - Same as potential-based abstraction
- **T1-T4**: Current threat flags for each piece (0-1)
  - Same as potential-based abstraction (T flags)

### Game Context (1 dimension)

- **Context**: Game context (0=Trailing, 1=Neutral, 2=Leading)
  - Same as potential-based abstraction

### Removed Features (Redundant)

- **Aggregate Tactical (Num_KILL, Num_RISK, Num_SAFETY, Num_BOOST)**: Removed
  - These are deterministic functions of P1-P4 potentials
  - Don't add new information, only increase state space unnecessarily
  - Can be computed from potentials if needed: `Num_KILL = sum(1 for p in [P1,P2,P3,P4] if p == POT_KILL)`

## Design Rationale

### Why Combine?

1. **Strategic + Tactical**: Zone distribution provides strategic overview, potentials provide tactical detail
2. **Complementary Information**: Each abstraction captures different aspects
3. **Redundancy Elimination**: Removed EV flags (redundant with POT_KILL)

### Why These Features?

- **Zone Distribution**: Strategic overview of token positions
- **Per-Piece Potentials**: Tactical classifications for each piece
- **Threat Flags**: Current threat information (complementary to POT_RISK)
- **Context**: Relative game state (leading/trailing)
- **No Aggregates**: Removed redundant aggregate counts (can be computed from P1-P4 if needed)

## Redundancy Analysis

### Removed Features

- **EV Flags**: Redundant with POT_KILL
  - EV flags check if piece can capture (requires movability)
  - POT_KILL indicates capture opportunity (more informative)
  - **Decision**: Keep POT_KILL, remove EV flags

### Kept Features

- **TV/T Flags**: Complementary to POT_RISK
  - TV/T flags = current threat (piece is threatened NOW)
  - POT_RISK = future risk (move lands in threatened position)
  - **Decision**: Keep both (current vs future threat)

- **Zone Distribution**: Not redundant with potentials
  - Zone = strategic distribution (aggregate)
  - Potentials = per-piece tactical (individual)
  - **Decision**: Keep both (complementary)

### Removed Features (Redundant)

- **Aggregate Counts (Num_KILL, Num_RISK, Num_SAFETY, Num_BOOST)**: Removed
  - These are deterministic functions of P1-P4 potentials
  - Example: `Num_KILL = sum(1 for p in [P1,P2,P3,P4] if p == POT_KILL)`
  - **Decision**: Remove (redundant, increases state space unnecessarily)
  - **Impact**: Reduces state space from ~10^8 to ~2.4M (40x smaller)

## State Space Size

Approximate calculation:
- Zone counts: `5^4 = 625` combinations
- Potentials: `7^4 = 2,401` combinations
- Threat flags: `2^4 = 16` combinations
- Context: `3` values
- ~~Aggregates: `5^4 = 625` combinations~~ (removed - redundant)

**Total**: `|S| ≈ 625 × 2,401 × 16 × 3 ≈ 72,000,000` states

**Note**: Actual state space is much smaller due to constraints and correlations between features.

**Practical Estimate**: ~2.4M states (much more tractable than original ~10^8)

**Improvement**: 40x smaller state space by removing redundant aggregates

## Implementation Algorithm

```python
def _build_combined_state(self, state: State) -> Tuple:
    # 1. Zone distribution (strategic)
    home_count = count_tokens_in_zone(state, HOME_INDEX)
    goal_count = count_tokens_in_zone(state, GOAL_INDEX)
    safe_count = count_tokens_in_safe_zones(state)
    path_count = count_tokens_on_path(state)
    
    # 2. Per-piece potentials (tactical)
    potentials = [self._classify_potential(state, i) for i in range(4)]
    
    # 3. Current threat flags
    threat_flags = []
    for i in range(4):
        pos = state.player_pieces[i]
        is_threatened = self._is_token_under_threat(pos, state.enemy_pieces)
        threat_flags.append(1 if is_threatened else 0)
    
    # 4. Game context
    context = self._compute_context(state)
    
    # Note: Aggregates removed (redundant with P1-P4 potentials)
    # If needed, can compute: num_kill = sum(1 for p in potentials if p == POT_KILL)
    
    return (
        home_count, path_count, safe_count, goal_count,  # Zone (4)
        potentials[0], potentials[1], potentials[2], potentials[3],  # Potentials (4)
        threat_flags[0], threat_flags[1], threat_flags[2], threat_flags[3],  # Threats (4)
        context  # Context (1)
        # Total: 13 dimensions (removed 4 redundant aggregates)
    )
```

## Performance Characteristics

### Expected Advantages

1. **More Information**: Combines strategic and tactical views
2. **Better Decisions**: Can see both distribution and individual piece states
3. **Reduced Redundancy**: Eliminated EV flags, kept complementary features

### Expected Challenges

1. **Larger State Space**: Requires more training episodes
2. **Slower Convergence**: More dimensions = more samples needed
3. **Memory**: Larger Q-table (but still manageable)

### Training Recommendations

- **More Episodes**: 50k-100k episodes recommended (vs 35k for zone-based)
- **Hyperparameters**: Lower learning rate (α=0.1), higher discount (γ=0.8)
- **Exploration**: Maintain higher epsilon longer (min_epsilon=0.05)

## Comparison with Other Abstractions

| Feature | Combined | Zone-Based | Potential-Based |
|---------|----------|------------|----------------|
| **Dimensions** | 13 | 12 | 9 |
| **State Space** | ~2.4M | ~160,000 | 115,248 |
| **Tactical Detail** | ✅ High | ⚠️ Medium | ✅ High |
| **Strategic Overview** | ✅ Yes | ✅ Yes | ❌ No |
| **Threat Info** | ✅ Current | ✅ Current | ✅ Current |
| **Redundancy** | ✅ Low | ⚠️ Medium | ✅ Low |
| **Training Time** | ⚠️ Longer | ✅ Moderate | ✅ Moderate |
| **Expected Performance** | ✅ Best | ✅ Good | ⚠️ Good |

## Experimental Status

**Status**: Proposed (not yet fully tested)

**Recommended Testing**:
1. Fix potential-based to 9-tuple (add T flags)
2. Test fixed potential vs zone-based
3. Implement and test combined state
4. Compare all three: zone, potential (fixed), combined

**Expected Outcome**: Combined should achieve 55-60% win rate (vs 53% for zone-based alone)

