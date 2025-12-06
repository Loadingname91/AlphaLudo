# Zone-Based State Abstraction (12-Tuple)

## Overview

The Zone-Based State Abstractor converts raw board states into a strategic 12-tuple:
`(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)`. This abstraction focuses on **strategic token distribution** across board zones and **per-piece vulnerability information**.

**Use Case**: Tabular Q-Learning (discrete state space)

**Reference**: Inspired by [Ludo_Game_AI repository](https://github.com/raffaele-aurucci/Ludo_Game_AI)

## State Tuple Structure

**Format**: `(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)`

### Zone Distribution (4 dimensions)

- **HOME**: Count of tokens in base/home (0-4)
- **PATH**: Count of tokens on main path, not in safe zones (0-4)
- **SAFE**: Count of tokens in safe zones (globes, home corridor, start) (0-4)
- **GOAL**: Count of tokens that reached goal (0-4)

### Vulnerability Flags (8 dimensions)

- **EV1-EV4**: Enemy vulnerable flags (0-1)
  - `EVi = 1` if piece *i* can capture an enemy with current dice roll
  - `EVi = 0` otherwise
- **TV1-TV4**: Token vulnerable flags (0-1)
  - `TVi = 1` if piece *i* is currently under threat (enemy can reach it in 1-6 steps)
  - `TVi = 0` otherwise

## Zone Classification

### Home Zone
- **Position**: `HOME_INDEX = 0`
- **Count**: Number of pieces at home

### Goal Zone
- **Position**: `GOAL_INDEX = 57`
- **Count**: Number of pieces that reached goal

### Safe Zones
- **Positions**: Globes (1, 9, 22, 35, 48), Home Corridor (52-56), Start (1)
- **Count**: Number of pieces in safe positions (excluding HOME and GOAL)

### Path Zone
- **Positions**: Main board (1-51) that are NOT in safe zones
- **Count**: Number of pieces on main path, exposed to capture

## Vulnerability Calculation

### Enemy Vulnerable (EV) Flags

```python
for piece_idx in range(4):
    pos = player_pieces[piece_idx]
    if pos in [HOME_INDEX, GOAL_INDEX]:
        ev_flags.append(0)
        continue
    
    # Check if this token can capture an enemy
    can_capture = False
    if is_piece_movable(state, piece_idx):
        next_pos = simulate_move(pos, state.dice_roll)
        can_capture = can_capture_enemy(next_pos, state.enemy_pieces)
    ev_flags.append(1 if can_capture else 0)
```

**Key Points**:
- Requires piece to be movable
- Checks if next position can capture enemy
- Similar to POT_KILL in potential-based abstraction

### Token Vulnerable (TV) Flags

```python
for piece_idx in range(4):
    pos = player_pieces[piece_idx]
    if pos in [HOME_INDEX, GOAL_INDEX]:
        tv_flags.append(0)
        continue
    
    # Check if this token is under threat
    is_threatened = is_token_under_threat(pos, state.enemy_pieces)
    tv_flags.append(1 if is_threatened else 0)
```

**Key Points**:
- Checks current position (not next position)
- Same logic as T flags in potential-based abstraction
- Indicates immediate danger

## State Space Size

The state space is approximately:
- Zone counts: `5^4 = 625` combinations (HOME+PATH+SAFE+GOAL = 4, so 0-4 each)
- EV flags: `2^4 = 16` combinations
- TV flags: `2^4 = 16` combinations

**Total**: `|S| ≈ 625 × 16 × 16 ≈ 160,000` states (varies based on constraints)

**Note**: Actual state space is smaller due to constraints (HOME+PATH+SAFE+GOAL ≤ 4)

## Implementation Notes

- **Zone Counting**: Count tokens in each zone, clamp to valid range (0-4)
- **Safe Zone Detection**: Use `is_safe_position()` to identify safe zones
- **Path Calculation**: Main board positions (1-51) that are not safe
- **Vulnerability Checks**: Run for each piece, skip HOME/GOAL positions
- **Coordinate Conversion**: Enemy pieces are in their own frames, handled by helper functions

## Performance Characteristics

- **State Space**: ~160,000 states (tractable for tabular Q-learning)
- **Information**: Strategic distribution + per-piece vulnerabilities
- **Strengths**: Strategic overview, compact representation, good generalization
- **Weaknesses**: Less tactical detail than potential-based (no POT_BOOST, POT_SAFETY classifications)

## Comparison with Other Abstractions

| Feature | Zone-Based | Potential-Based | Combined |
|---------|------------|----------------|----------|
| **Dimensions** | 12 | 9 | 17 |
| **State Space** | ~160,000 | 115,248 | ~10^8 |
| **Tactical Detail** | ⚠️ Medium | ✅ High | ✅ High |
| **Strategic Overview** | ✅ Yes | ❌ No | ✅ Yes |
| **Threat Info** | ✅ Current | ✅ Current | ✅ Current |
| **Use Case** | Tabular Q | Tabular Q | Tabular Q |

## Experimental Results

Based on self-play training experiments:
- **Phase 1 (vs Random)**: ~51-54% win rate
- **Phase 2 (Self-Play)**: ~53-55% win rate
- **Improvement**: +1.5-2% from self-play training

Zone-based abstraction has shown **better performance** than potential-based (5-tuple variant) in experiments, likely due to:
1. Strategic overview (token distribution)
2. Compact representation (better sample efficiency)
3. Good generalization across similar board states

