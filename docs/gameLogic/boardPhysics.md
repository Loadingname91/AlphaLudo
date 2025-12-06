# Board Physics and Game Logic

## Overview

This document describes the core game mechanics, special tiles, threat detection, capture logic, and safety zones in Ludo. All physics calculations must handle edge cases correctly for reliable agent training.

## Board Constants

### Position Indices

- **HOME_INDEX**: 0 (piece at home, cannot move unless dice=6)
- **START_INDEX**: 1 (starting position after exiting home)
- **GOAL_INDEX**: 57 (final position, piece has won)
- **HOME_STRETCH_INDEXS**: [52, 53, 54, 55, 56] (home corridor, safe zone)
- **GLOB_INDEXS**: [9, 22, 35, 48] (safe globes)
- **STAR_INDEXS**: [5, 12, 18, 25, 31, 38, 44, 51] (star jump tiles)
- **ENEMY_GLOBS**: [14, 27, 40] (enemy spawn positions, dangerous zones)

### Special Rules

- **DICE_MOVE_OUT_OF_HOME**: 6 (required to exit home)
- **Main Board Loop**: 52 positions (1-52, wraps around)

## Core Mechanics

### Circular Distance Calculation

**Purpose**: Calculate distance between two positions on the circular main board, handling wrap-around.

**Formula**:
```python
if target_pos > GOAL_INDEX or source_pos > GOAL_INDEX or \
   source_pos == HOME_INDEX or target_pos == HOME_INDEX:
    return target_pos - source_pos  # Linear distance for special zones
else:
    dist = target_pos - source_pos
    if dist < 0:
        dist += 52  # Wrap around
    return dist
```

**Edge Cases**:
- **Special Zones**: Home (0), Goal (57), positions > 57 → Use linear distance (no wrap-around)
- **Wrap-Around**: If `dist < 0`, add 52 to get circular distance
- **Example**: Position 5 to position 50 → dist = 5-50 = -45 → -45+52 = 7 steps

### Move Simulation

**Purpose**: Predict where a piece will land given current position and dice roll.

**Algorithm**:
```python
1. If current_pos == HOME_INDEX:
   - If dice_roll == 6: return START_INDEX (1)
   - Else: return HOME_INDEX (stuck)

2. If current_pos == GOAL_INDEX:
   - return GOAL_INDEX (already finished)

3. Calculate raw position: predicted_pos = current_pos + dice_roll

4. Goal bounce: If predicted_pos > GOAL_INDEX:
   - overshoot = predicted_pos - GOAL_INDEX
   - predicted_pos = GOAL_INDEX - overshoot

5. Star jump: If predicted_pos in STAR_INDEXS:
   - idx = STAR_INDEXS.index(predicted_pos)
   - If predicted_pos == 51 (last star): return GOAL_INDEX
   - Else: return STAR_INDEXS[idx+1] (next star)
```

**Edge Cases**:

#### Home Exit
- **Scenario**: Piece at home (0), dice roll is 6
- **Handling**: Return `START_INDEX` (1)
- **Other Dice**: Return `HOME_INDEX` (0), piece stays home

#### Goal Bounce
- **Scenario**: Move overshoots goal (e.g., position 56, dice 3 → predicted 59)
- **Handling**: `overshoot = 59 - 57 = 2`, `predicted_pos = 57 - 2 = 55`
- **Rationale**: Piece bounces back from goal

#### Star Jumps
- **Scenario**: Land exactly on a star tile
- **Handling**: Jump to next star in sequence
- **Last Star (51)**: Jump to `GOAL_INDEX` (57)
- **Index Error**: Should never happen (51 handled separately), but raise ValueError if it does

#### Star Jump Chains
- **Scenario**: Multiple consecutive star jumps
- **Handling**: Prevented by game rules (can only jump once per move)
- **Rationale**: Documented for completeness, shouldn't occur

## Special Tiles

### Globes (Safe Zones)
- **Positions**: [9, 22, 35, 48]
- **Properties**: 
  - Safe from capture
  - Cannot capture enemies here
  - Multiple pieces can occupy (no blockade formation)

### Stars (Jump Tiles)
- **Positions**: [5, 12, 18, 25, 31, 38, 44, 51]
- **Properties**:
  - Landing on star triggers jump to next star
  - Last star (51) jumps to goal (57)
  - Extra movement (boost)

### Home Corridor (Goal Path)
- **Positions**: [52, 53, 54, 55, 56]
- **Properties**:
  - Safe zone (cannot be captured)
  - One-way path to goal
  - Only accessible from main board (position 51 or star jump)

### Goal
- **Position**: 57
- **Properties**:
  - Final destination
  - Piece cannot move from goal
  - Winning condition (all 4 pieces at goal)

## Threat Detection

### Is Threatened

**Purpose**: Check if a piece is threatened by enemies (1-6 steps behind).

**Algorithm**:
```python
1. If is_safe_position(piece_pos): return False (immediate bypass)

2. For each enemy in enemy_pieces:
   - Skip enemies at HOME or GOAL
   - Calculate distance: dist = get_circular_distance(piece_pos, enemy_pos)
   - If 1 <= dist <= 6: return True (threatened)

3. Check enemy spawn danger: is_enemy_spawn_danger(piece_pos)
   - If True: return True
```

**Edge Cases**:

#### Safe Position Bypass
- **Scenario**: Piece is in safe zone
- **Handling**: Return `False` immediately (no threat calculation)
- **Rationale**: Safe positions cannot be captured

#### Enemy Spawn Danger
- **Scenario**: Position is 1-6 steps ahead of enemy globes (14, 27, 40)
- **Handling**: Check `1 <= (pos - spawn) <= 6` for each spawn
- **Rationale**: Enemy pieces spawn at globes, positions ahead are risky

#### Circular Distance for Threats
- **Scenario**: Enemy behind piece on circular board
- **Handling**: Use `get_circular_distance()` for proper wrap-around calculation
- **Rationale**: Enemies can be behind even if position number is higher (wrap-around)

## Capture Logic

### Can Capture

**Purpose**: Check if a move can capture an enemy piece.

**Algorithm**:
```python
1. Simulate move: predicted_pos = simulate_move(from_pos, dice_roll)

2. Check safe zones: If predicted_pos in [GOAL, HOME] + GLOB_INDEXS:
   - return False (cannot capture in safe zones)

3. Check enemy at position: enemy_at_pos, _ = get_enemy_at_pos(predicted_pos, enemy_pieces)
   - If enemy_at_pos != NO_ENEMY: return True
```

**Edge Cases**:

#### Safe Zone Exceptions
- **Cannot Capture In**:
  - Goal (57)
  - Home (0)
  - Globes [9, 22, 35, 48]
- **Can Capture On**:
  - Stars (before jump)
  - Start (1) - special case, can capture from start
  - Main board positions

#### Coordinate System Conversion
- **Scenario**: Enemy pieces are in their own coordinate frames
- **Handling**: Use `get_enemy_at_pos()` which handles coordinate conversion
- **Rationale**: Each player has their own coordinate system, conversion is automatic

#### Star Jump Capture
- **Scenario**: Landing on star before jump
- **Handling**: Capture happens at final destination (after jump), not at star
- **Rationale**: Ludopy simplifies this - capture at final position

## Safety Zones

### Is Safe Position

**Returns True if position is**:
- Home (0)
- Start (1)
- Globes [9, 22, 35, 48]
- Home Corridor [52, 53, 54, 55, 56]
- Goal (57)

**Properties**:
- Cannot be captured
- Multiple pieces can occupy (no blockade restrictions)
- Safe from enemy threats

## Blockade Formation

### Can Form Blockade

**Purpose**: Check if moving to a position forms a blockade (2+ pieces at same position).

**Algorithm**:
```python
1. If new_pos in [HOME, GOAL]: return False (cannot form blockade)

2. Check if any other piece is at new_pos:
   - For each piece_idx != moving_piece_idx:
     - If player_pieces[piece_idx] == new_pos: return True
```

**Edge Cases**:

#### Home/Goal Exceptions
- **Scenario**: Moving to home or goal
- **Handling**: Return `False` (cannot form blockade)
- **Rationale**: Special positions, blockade rules don't apply

#### Blockade Detection
- **Scenario**: 2+ pieces at same position
- **Handling**: Count pieces at position, return `True` if count >= 2
- **Rationale**: Blockades provide defensive advantage

### Piece Is In Blockade

**Purpose**: Check if a specific piece is part of a blockade.

**Algorithm**:
```python
1. Get piece position: pos = player_pieces[piece_idx]

2. If pos in [HOME, GOAL]: return False

3. Count pieces at position: count = sum(1 for p in player_pieces if p == pos)

4. Return count >= 2
```

### Enemy Blockade At

**Purpose**: Check if enemies have a blockade at a specific position.

**Algorithm**:
```python
For each opponent_pieces in enemy_pieces:
    count = sum(1 for p in opponent_pieces if p == pos)
    if count >= 2: return True
return False
```

## Strategic Helpers

### Distance to Goal

**Formula**:
```python
if pos == HOME_INDEX: return 57
if pos >= GOAL_INDEX: return 0
return GOAL_INDEX - pos
```

### Get Least Advanced Piece

**Purpose**: Find the piece furthest from goal (least progress).

**Algorithm**:
```python
1. Filter pieces not at home/goal: candidates = [(idx, pos) for idx, pos in enumerate(pieces) if pos not in [HOME, GOAL]]

2. If no candidates: return 0 (default)

3. Return idx of piece with minimum position (closest to home)
```

### Enemy Spawn Danger

**Purpose**: Check if position is 1-6 steps ahead of enemy spawn points.

**Algorithm**:
```python
For each spawn in ENEMY_GLOBS [14, 27, 40]:
    if 1 <= (pos - spawn) <= 6:
        return True
return False
```

**Rationale**: Enemy pieces spawn at globes, positions immediately ahead are risky.

### Count Helpers

- **count_pieces_at_home**: Count pieces at position 0
- **count_pieces_at_goal**: Count pieces at position 57
- **count_pieces_on_goal_path**: Count pieces in [52, 53, 54, 55, 56]

## Edge Cases Summary

### Circular Distance
- Wrap-around calculation: `dist < 0 → dist + 52`
- Special zones: Home, Goal, positions > 57 → Linear distance
- Boundary: Position 52 to 1 → Wrap-around (1 step, not 51)

### Move Simulation
- Home exit: Only with dice=6, returns START_INDEX
- Goal bounce: Overshoot calculation `GOAL_INDEX - overshoot`
- Star jump: Last star (51) → GOAL_INDEX, others → next star
- Index error: Should never happen (51 handled), but raise ValueError

### Threat Detection
- Safe position bypass: Return False immediately
- Enemy spawn danger: Check positions 1-6 ahead of spawns
- Circular wrap-around: Use `get_circular_distance()` for proper calculation

### Capture Logic
- Safe zone exceptions: Cannot capture in globes, goal, home
- Coordinate conversion: Handled by `get_enemy_at_pos()`
- Star jump: Capture at final destination, not at star

### Blockade Formation
- Home/goal exceptions: Cannot form blockade
- Count threshold: 2+ pieces required
- Multiple blockades: Can have multiple blockades at different positions

