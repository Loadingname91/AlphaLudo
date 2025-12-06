# Orthogonal State Abstraction (31-dim)

## Overview

The Orthogonal State Abstractor converts raw Ludo game state into a 31-dimensional feature vector. The features are "orthogonal" in that they represent distinct, non-overlapping aspects of the game state.

## Feature Vector Structure

**Total Dimensions**: 31

### Per-Piece Features (20 dims = 5 features × 4 pieces)

For each of the 4 pieces, extract 5 features:

1. **Normalized Progress** (0.0-1.0)
   - Formula: `piece_pos / 57.0`
   - Range: 0.0 (home) to 1.0 (goal)
   - Represents how far the piece has progressed

2. **Is Safe** (0 or 1)
   - Binary indicator: 1 if piece is in safe position, 0 otherwise
   - Safe positions: Stars, Globes, Start, Home, Home Corridor, Goal
   - Formula: `1 if piece_pos in safe_positions else 0`

3. **In Home Corridor** (0 or 1)
   - Binary indicator: 1 if piece is in home corridor (positions 52-56), 0 otherwise
   - Formula: `1 if piece_pos in HOME_STRETCH_INDEXS else 0`

4. **Threat Distance** (0.0-1.0)
   - Normalized distance to nearest threatening enemy (1-6 steps behind)
   - Formula: `min(1.0, min_threat_distance / 6.0)`
   - If no threats: 1.0 (safe)
   - If threats exist: minimum distance / 6.0
   - Only considers enemies 1-6 steps behind (using circular distance)

5. **Kill Opportunity** (0 or 1)
   - Binary indicator: 1 if piece can capture an enemy with current dice roll, 0 otherwise
   - Formula: `1 if can_capture(piece_pos, dice_roll, enemy_pieces) else 0`

### Global Features (11 dims)

21. **Relative Progress** (-1.0 to 1.0)
   - Normalized difference between agent's average position and enemy average position
   - Formula: `(my_avg_pos - enemy_avg_pos) / 57.0`
   - Positive: Agent ahead, Negative: Agent behind

22. **Pieces in Yard** (0.0-1.0)
   - Fraction of pieces still at home
   - Formula: `count(pieces == HOME) / 4.0`
   - Range: 0.0 (all out) to 1.0 (all at home)

23. **Pieces Scored** (0.0-1.0)
   - Fraction of pieces that reached goal
   - Formula: `count(pieces == GOAL) / 4.0`
   - Range: 0.0 (none scored) to 1.0 (all scored)

24. **Enemy Scored** (0.0-1.0)
   - Fraction of enemy pieces that reached goal
   - Formula: `count(all_enemy_pieces == GOAL) / 12.0`
   - Range: 0.0 (no enemies scored) to 1.0 (all enemies scored)

25. **Max Kill Potential** (0.0-1.0)
   - Fraction of pieces that can capture enemies
   - Formula: `min(1.0, count(can_capture) / 4.0)`
   - Range: 0.0 (no kill opportunities) to 1.0 (all pieces can kill)

26-31. **Dice Roll One-Hot** (6 dims)
   - One-hot encoding of current dice roll (1-6)
   - Formula: `[0.0] * 6`, then `dice_one_hot[dice_roll - 1] = 1.0`

## Edge Cases

### Empty Enemy Pieces
- **Scenario**: `state.enemy_pieces` is empty or all enemies at home/goal
- **Threat Distance Handling**: 
  - If no threats found: `threat_distance = 1.0` (safe)
  - Skip enemies at HOME or GOAL in threat calculation
- **Relative Progress Handling**: 
  - If `all_enemy_positions` is empty: `enemy_avg_pos = 0.0`
- **Enemy Scored Handling**: 
  - Count across all enemy lists, handle empty lists gracefully

### All Pieces at Home/Goal
- **Normalized Progress**: 
  - Home (0): `0 / 57.0 = 0.0` ✓
  - Goal (57): `57 / 57.0 = 1.0` ✓
- **Pieces in Yard**: All at home → `4 / 4.0 = 1.0` ✓
- **Pieces Scored**: All at goal → `4 / 4.0 = 1.0` ✓

### Invalid Piece Positions
- **Boundary Checks**: Positions should be 0-57 (home to goal)
- **Out of Bounds**: Positions < 0 or > 57 should not occur, but if they do:
  - Progress normalization: Clamp or handle gracefully
  - Safe position check: May return False for invalid positions

### Threat Distance Calculation
- **Circular Wrap-Around**: 
  - Use `get_circular_distance(piece_pos, enemy_pos)` for proper calculation
  - Handles wrap-around on main board (52 positions)
- **Safe Position Handling**: 
  - If piece is in safe position, threat calculation still runs but should return 1.0 (no threats)
  - However, safe positions are checked separately in "Is Safe" feature
- **Multiple Threats**: 
  - Collect all threat distances (1-6 steps)
  - Take minimum: `min_threat = min(threat_distances)`
  - Normalize: `min(1.0, min_threat / 6.0)`

### Kill Opportunity
- **Coordinate System Conversion**: 
  - Enemy pieces are in their own coordinate frames
  - `can_capture()` handles coordinate conversion via `get_enemy_at_pos()`
- **Safe Zone Exceptions**: 
  - Cannot capture in globes, goal, home (handled in `can_capture()`)

### Progress Normalization
- **Home Position**: `0 / 57.0 = 0.0` ✓
- **Goal Position**: `57 / 57.0 = 1.0` ✓
- **Main Board**: `pos / 57.0` (0.0 to 1.0) ✓
- **Home Corridor**: Positions 52-56 normalized as `pos / 57.0` (0.91 to 0.98)

### Dice Roll One-Hot
- **Invalid Dice Roll**: 
  - If `dice_roll < 1` or `dice_roll > 6`: All zeros (no one-hot)
  - Should not happen in normal gameplay

## Implementation Notes

- **Safe Positions List**: `stars + globes + [home, START]`
- **Home Corridor**: `HOME_STRETCH_INDEXS = [52, 53, 54, 55, 56]`
- **Threat Distance**: Only considers enemies 1-6 steps behind (positive distance)
- **Enemy Piece Iteration**: Nested loop over 3 enemies, each with 4 pieces
- **Feature Order**: Per-piece features first (20 dims), then global features (11 dims)

