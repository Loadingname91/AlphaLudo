# Augmented Raw State Abstraction (90-dim)

## Overview

The Augmented Raw State Abstractor constructs a 90-dimensional state vector that is isomorphic to the raw board but semantically enriched with physics rules and local tactical dynamics. This representation preserves the full board structure while adding tactical information.

## Feature Vector Structure

**Total Dimensions**: 90

### Global Context (10 dims)

1-6. **Dice Roll One-Hot** (6 dims)
   - One-hot encoding of current dice roll (1-6)
   - Formula: `[0.0] * 6`, then `dice_vec[dice_roll - 1] = 1.0`
   - Edge case: If `dice_roll < 1` or `> 6`, all zeros

7-10. **Active Player One-Hot** (4 dims)
   - One-hot encoding of current player (0-3)
   - Formula: `[0.0] * 4`, then `player_vec[current_player] = 1.0`
   - Edge case: If `current_player < 0` or `>= 4`, all zeros

### Token Data (80 dims = 16 tokens × 5 features)

**Token Order**: Player 0 tokens, Player 1 tokens, Player 2 tokens, Player 3 tokens (fixed order)

**Features per Token** (5 dims):

#### Current Player Tokens

1. **Normalized Position** (0.0-1.0)
   - Home (0): `0.0`
   - Goal (57): `1.0`
   - Main board: `pos / 57.0`

2. **Safe Indicator** (0.0 or 1.0)
   - Binary: 1.0 if `is_safe_position(pos)`, else 0.0
   - Safe positions: Home, Start, Globes, Home Corridor, Goal

3. **Kill Opportunity** (0.0 or 1.0)
   - Binary: 1.0 if `can_capture(pos, dice_roll, enemy_pieces)`, else 0.0

4. **Threat Indicator** (0.0 or 1.0)
   - Binary: 1.0 if `is_threatened(pos, enemy_pieces)`, else 0.0

5. **Opponent Flag** (0.0)
   - Always 0.0 for current player tokens

#### Enemy Tokens

1. **Normalized Position** (0.0-1.0)
   - Same normalization as current player: Home→0.0, Goal→1.0, else `pos/57.0`

2. **Goal Indicator** (0.0 or 1.0)
   - Binary: 1.0 if `is_at_goal(pos)`, else 0.0

3. **Reserved Field 1** (0.0)
   - Always 0.0 (reserved for future use)

4. **Reserved Field 2** (0.0)
   - Always 0.0 (reserved for future use)

5. **Opponent Flag** (1.0)
   - Always 1.0 for enemy tokens

## Board State Reconstruction

### Coordinate System Handling

The state object provides:
- `state.player_pieces`: Pieces of `state.current_player` (in current player's coordinate frame)
- `state.enemy_pieces`: List of 3 enemy piece lists (each in their own coordinate frame)

**Reconstruction Logic**:
```python
board_state = {}
current_pid = state.current_player
board_state[current_pid] = state.player_pieces

for i, enemy_pieces_list in enumerate(state.enemy_pieces):
    # Enemy 0 is (current + 1) % 4
    enemy_pid = (current_pid + 1 + i) % 4
    board_state[enemy_pid] = enemy_pieces_list
```

**Fixed Order Iteration**: Always iterate players 0, 1, 2, 3 in that order for consistent feature vector.

## Edge Cases

### Missing Player Pieces
- **Scenario**: Player has fewer than 4 pieces on board
- **Handling**: `tokens = board_state.get(pid, [])` returns empty list if player not found
- **Result**: Fewer than 16 tokens in feature vector (still valid, just shorter)
- **Note**: Should always have exactly 4 pieces per player, but handle gracefully

### Coordinate System Conversion
- **Scenario**: Enemy pieces are in their own coordinate frames
- **Handling**: 
  - Enemy positions are already converted by ludopy when provided in `state.enemy_pieces`
  - Feature extraction uses positions as-is (no additional conversion needed)
  - Kill/threat detection uses `can_capture()` and `is_threatened()` which handle coordinate conversion

### Feature Extraction Edge Cases

#### Home/Goal Position Normalization
- **Home (0)**: `norm_pos = 0.0` (explicit check)
- **Goal (57)**: `norm_pos = 1.0` (explicit check)
- **Main Board**: `norm_pos = float(pos) / 57.0` (normal division)

#### Reserved Fields
- **Current Player**: Reserved fields not used (only 5 features, all meaningful)
- **Enemy Tokens**: Fields 3 and 4 are reserved (always 0.0)
- **Rationale**: Avoid misusing local coordinates for detailed physics on enemy tokens

### Board State Reconstruction Edge Cases

#### Current Player and Enemy Ordering
- **Current Player**: `state.current_player` (0-3)
- **Enemy Ordering**: `state.enemy_pieces[0]` is `(current + 1) % 4`, `[1]` is `(current + 2) % 4`, etc.
- **Fixed Order**: Always iterate players 0, 1, 2, 3 for consistent feature vector

#### Missing Players
- **Scenario**: Ghost players (players not in game)
- **Handling**: `board_state.get(pid, [])` returns empty list
- **Result**: No tokens for that player (feature vector may be shorter)

### Dice Roll One-Hot Edge Cases
- **Invalid Dice**: If `dice_roll < 1` or `> 6`, all zeros (no one-hot)
- **Should Not Happen**: Normal gameplay always has dice 1-6

### Active Player One-Hot Edge Cases
- **Invalid Player**: If `current_player < 0` or `>= 4`, all zeros (no one-hot)
- **Should Not Happen**: Normal gameplay always has player 0-3

## Implementation Notes

- **Token Iteration**: Always iterate players 0, 1, 2, 3 in fixed order
- **Feature Order**: Global context (10) → Token data (80)
- **Token Features**: 5 features per token, in order: Position, Safe/Kill/Threat indicators, Opponent flag
- **Current vs Enemy**: Different feature sets for current player (tactical) vs enemies (positional only)
- **Coordinate Frames**: Enemy positions are already in correct coordinate frame when provided

