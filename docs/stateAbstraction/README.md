# State Abstraction Documentation

This directory contains documentation for all state abstraction methods used in the Ludo RL agent.

## Tabular Q-Learning Abstractions

### 1. [Potential-Based State](potentialBasedState.md) (9-Tuple)
- **Format**: `(P1, P2, P3, P4, Context, T1, T2, T3, T4)`
- **State Space**: 115,248 states
- **Focus**: Tactical per-piece detail + game context
- **Use Case**: Tabular Q-Learning
- **Status**: ✅ Implemented (needs T flags fix)

### 2. [Zone-Based State](zoneBasedState.md) (12-Tuple)
- **Format**: `(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)`
- **State Space**: ~160,000 states
- **Focus**: Strategic token distribution + vulnerabilities
- **Use Case**: Tabular Q-Learning
- **Status**: ✅ Implemented and tested

### 3. [Combined State](combinedState.md) (13-Tuple)
- **Format**: `(HOME, PATH, SAFE, GOAL, P1-P4, T1-T4, Context)`
- **State Space**: ~2.4M states (optimized - removed redundant aggregates)
- **Focus**: Strategic + Tactical (best of both)
- **Use Case**: Tabular Q-Learning
- **Status**: ✅ Implemented and optimized

## Function Approximation Abstractions

### 4. [Orthogonal State](orthogonalState.md) (31-Dim)
- **Format**: 31-dimensional continuous feature vector
- **Focus**: Per-piece features + global features
- **Use Case**: DQN / Function Approximation
- **Status**: ✅ Documented (for DQN implementation)

### 5. [Augmented Raw State](augmentedRawState.md) (90-Dim)
- **Format**: 90-dimensional feature vector
- **Focus**: Full board representation with tactical indicators
- **Use Case**: DQN / Function Approximation
- **Status**: ✅ Documented (for DQN implementation)

## Quick Comparison

| Abstraction | Dimensions | State Space | Use Case | Status |
|------------|-----------|-------------|----------|--------|
| **Potential-Based** | 9 | 115,248 | Tabular Q | ✅ Implemented |
| **Zone-Based** | 12 | ~160,000 | Tabular Q | ✅ Implemented |
| **Combined** | 13 | ~2.4M | Tabular Q | ✅ Implemented (optimized) |
| **Orthogonal** | 31 | Continuous | DQN | ✅ Documented |
| **Augmented Raw** | 90 | Continuous | DQN | ✅ Documented |

## Recommendations

### For Tabular Q-Learning:
1. **Zone-Based**: Best balance of performance and efficiency (recommended)
2. **Potential-Based**: Fixed with T flags (9-tuple) - good tactical detail
3. **Combined**: Optimized 13-tuple (removed redundant aggregates) - combines strategic + tactical

### For Function Approximation (DQN):
1. **Use Orthogonal**: 31-dim features (good balance)
2. **Use Augmented Raw**: 90-dim features (maximum information)

## Related Documentation

- [State Abstraction Gap Analysis](../../docs/STATE_ABSTRACTION_GAP_ANALYSIS.md) - Analysis of missing features
- [Combined State Analysis](../../docs/COMBINED_STATE_ANALYSIS.md) - Detailed redundancy analysis
- [Tabular Q-Learning Methodology](../../docs/agents/tabularQLearningMethodology.md) - Agent implementation details

