# Potential-Based State Abstraction (9-Tuple)

## Overview

The Potential-Based State Abstractor converts raw board states into a tactical 9-tuple:
`(P1, P2, P3, P4, Context, T1, T2, T3, T4)`. This abstraction is designed for tabular
Q-learning, focusing on **tactical move potentials** and **explicit per-piece threat information**.

**Use Case**: Tabular Q-Learning (discrete state space)

## State Tuple Structure

**Format**: `(P1, P2, P3, P4, Context, T1, T2, T3, T4)`

- **P1-P4**: Potential classification for each piece (0-6)
- **Context**: Game context (0=Trailing, 1=Neutral, 2=Leading)
- **T1-T4**: Binary threat flags for each piece (0 or 1)
  - `Ti = 1` if piece *i* is currently in a position that can be captured by any enemy
    in 1–6 steps on their next move
  - `Ti = 0` otherwise

### State Space Size

- Potentials: `P1-P4 ∈ {0..6}` → `7^4` combinations
- Context: `Context ∈ {0,1,2}` → `× 3`
- Threat flags: `T1-T4 ∈ {0,1}` → `× 2^4`

**Total**: `|S| = 7^4 × 3 × 2^4 = 115,248` states

This enables the agent to differentiate between "this move is risky" (POT_RISK) and "this piece is already under threat" (Ti=1) at the state level.

## Potential Classifications

| Potential | Value | Description |
|-----------|-------|-------------|
| POT_NULL | 0 | Piece cannot move (home without dice=6, goal, not in movable_pieces) |
| POT_NEUTRAL | 1 | Normal move with no special outcome |
| POT_RISK | 2 | Move lands in threatened position (enemy 1-6 steps behind) |
| POT_BOOST | 3 | Star jump (extra movement via star tile) |
| POT_SAFETY | 4 | Move lands in safe zone (globe, home corridor, start) |
| POT_KILL | 5 | Move captures an enemy piece |
| POT_GOAL | 6 | Move reaches goal (winning move) |

## Potential Classification Algorithm

For each piece, simulate the move and classify the outcome:

```python
next_pos = simulate_move(current_pos, dice_roll)

# Priority order (highest first):
1. If next_pos == GOAL_INDEX: return POT_GOAL
2. If can_capture(current_pos, dice, enemy_pieces): return POT_KILL
3. If next_pos in HOME_CORRIDOR or SAFE_GLOBES: return POT_SAFETY
4. If star_jump_detected: return POT_BOOST
5. If is_threatened(next_pos, enemy_pieces): return POT_RISK
6. Else: return POT_NEUTRAL
```

## Context Calculation

### Weighted Equity Scoring

**Score Formula**:
```
Score = (Goal × 100) + (Corridor × 50) + (Safe × 10) + (Distance)
```

**Scoring Details**:
- **Goal position (57)**: +100 points
- **Home Corridor (52-56)**: +50 + position (e.g., 52 → 102, 56 → 106)
- **Safe Globes/Start (1, 9, 22, 35, 48)**: +10 + position (e.g., 1 → 11, 48 → 58)
- **Main board**: +position (raw distance, e.g., 10 → 10, 40 → 40)
- **Home (0)**: +0 points

### Context Thresholds

```python
my_score = get_weighted_score(player_pieces)
max_opponent_score = max(get_weighted_score(enemy) for enemy in enemy_pieces)
gap = my_score - max_opponent_score

if gap < -20:
    return CONTEXT_TRAILING  # 0
elif gap > 20:
    return CONTEXT_LEADING   # 2
else:
    return CONTEXT_NEUTRAL   # 1
```

**Context Values**:
- **CONTEXT_TRAILING (0)**: Panic Mode (gap < -20)
- **CONTEXT_NEUTRAL (1)**: Balanced Race (-20 <= gap <= 20)
- **CONTEXT_LEADING (2)**: Lockdown Mode (gap > 20)

## Threat Flags (T1-T4)

Threat flags indicate **current threat** (piece is under threat NOW), as opposed to POT_RISK which indicates **future risk** (move lands in threatened position).

**Calculation**:
```python
for piece_idx in range(4):
    pos = state.player_pieces[piece_idx]
    is_threatened = is_token_under_threat(pos, state.enemy_pieces)
    threat_flags.append(1 if is_threatened else 0)
```

**Key Difference**:
- **T flags**: Current position is threatened
- **POT_RISK**: Next position would be threatened

## Implementation Notes

- **Potential Order**: Check in priority order (GOAL > KILL > SAFETY > BOOST > RISK > NEUTRAL)
- **Context Calculation**: Calculate once per state, reuse for all pieces
- **Movable Pieces Check**: Always check `state.movable_pieces` before classifying potential
- **Threat Flags**: Compute `Ti` using `is_token_under_threat()` on **current** positions
- **Tuple Return**: Ensure exactly 4 potentials + 1 context + 4 threat flags = 9-tuple

## Performance Characteristics

- **State Space**: 115,248 states (tractable for tabular Q-learning)
- **Information**: Tactical per-piece detail + game context
- **Strengths**: Fine-grained tactical awareness, context-adaptive
- **Weaknesses**: No strategic overview (token distribution), larger state space than zone-based

## Comparison with Other Abstractions

| Feature | Potential-Based | Zone-Based | Combined |
|---------|---------------|------------|----------|
| **Dimensions** | 9 | 12 | 17 |
| **State Space** | 115,248 | ~10^6 | ~10^8 |
| **Tactical Detail** | ✅ High | ⚠️ Medium | ✅ High |
| **Strategic Overview** | ❌ No | ✅ Yes | ✅ Yes |
| **Threat Info** | ✅ Current | ✅ Current | ✅ Current |
| **Use Case** | Tabular Q | Tabular Q | Tabular Q |

