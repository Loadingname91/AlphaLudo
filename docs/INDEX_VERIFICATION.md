# Index Verification - User Table vs Ludopy Library

## Ludopy's Official Index System (from player.py)

```python
HOME_INDEX = 0
START_INDEX = 1
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]
HOME_AREAL_INDEXS = [52, 53, 54, 55, 56]  # Home Stretch
GOAL_INDEX = 57
GLOB_INDEXS = [9, 22, 35, 48]  # ⚠️ NOTE: Does NOT include index 1
TOTAL_NUMBER_OF_TAILES = 58  # Indices 0-57
```

## Comparison with User's Table

| Feature | User's Table | Ludopy Library | Status |
|---------|--------------|----------------|--------|
| **Home (Yard)** | 0 | 0 | ✅ **CORRECT** |
| **Start Tile** | 1 | 1 | ✅ **CORRECT** |
| **Stars** | 5, 12, 18, 25, 31, 38, 44, 51 | 5, 12, 18, 25, 31, 38, 44, 51 | ✅ **CORRECT** |
| **Globes** | 1, 9, 22, 35, 48 | 9, 22, 35, 48 | ⚠️ **PARTIALLY WRONG** |
| **Choke Point** | 48 | 48 (is a globe) | ✅ **CORRECT** |
| **Home Entry** | 51 | 51 (last star) | ✅ **CORRECT** |
| **Home Stretch** | 52-56 | 52, 53, 54, 55, 56 | ✅ **CORRECT** |
| **Goal** | 57 | 57 | ✅ **CORRECT** |

## ⚠️ Key Issue: Index 1 (Start Tile)

### User's Table Says:
- **Globes**: 1, 9, 22, 35, 48

### Ludopy Actually Has:
- **Globes**: 9, 22, 35, 48 (NO index 1)
- **Start**: 1 (separate tile type: `TAILE_START`)

### Why the Confusion?

**Start tiles ARE functionally safe** (like globes), but ludopy tracks them differently:

```python
TAILE_START = 2   # Index 1 is marked as this
TAILE_GLOB = 3    # Indices 9, 22, 35, 48 are marked as this
```

**Visual representation**: The visualizer draws globe icons at ALL start positions (1, 14, 27, 40), which is why it might seem like index 1 is a globe.

**Functionally**: Start tiles and globes behave the same way:
- Both are SAFE positions (no captures)
- Multiple pieces can occupy them
- Protected from enemies

### Additional Globe Positions

From each player's perspective, OTHER players' start positions appear as "enemy globes":

```python
ENEMY_1_GLOB_INDX = 14  # Player 1's start (Yellow)
ENEMY_2_GLOB_INDX = 27  # Player 2's start (Blue)  
ENEMY_3_GLOB_INDX = 40  # Player 3's start (Pink)
```

These are marked as special types: `TAILE_ENEMY_1_GLOB`, `TAILE_ENEMY_2_GLOB`, `TAILE_ENEMY_3_GLOB`

## Corrected Table

| Feature | Indices | Notes |
|---------|---------|-------|
| **Home (Yard)** | 0 | Piece is off the board. Needs a 6 to move to 1. |
| **Start Tile** | 1 | High traffic area but SAFE (functionally like a globe). |
| **Stars** | 5, 12, 18, 25, 31, 38, 44, 51 | Landing here jumps to the next star. Index 51 can jump to Goal (57)! |
| **Globes (Safe)** | 9, 22, 35, 48 | True globe positions. Index 48 is the choke point. |
| **Enemy Start Globes** | 14, 27, 40 | Other players' start positions (appear as globes to you). |
| **All Safe Positions** | 1, 9, 14, 22, 27, 35, 40, 48 | All positions where pieces cannot be captured. |
| **Choke Point** | 48 | The globe right before home entry. Great for blockades. |
| **Home Entry** | 51 | Last star on main track. Must pass here to enter home stretch. |
| **Home Stretch** | 52, 53, 54, 55, 56 | Safe Zone. No enemies can enter. |
| **Goal** | 57 | The win condition. |

## Summary

### ✅ Correct Indices:
- Home: 0
- Start: 1
- Stars: 5, 12, 18, 25, 31, 38, 44, 51
- Home Stretch: 52-56
- Goal: 57

### ⚠️ Needs Clarification:
**Globes**: Should be **9, 22, 35, 48** (not including 1)

**However**, index 1 IS functionally safe (no captures allowed), it's just categorized as a "START" tile rather than a "GLOBE" tile in ludopy's internal system.

### Recommendation:

**For game strategy**, treat these as ALL safe positions:
- **Your start**: 1
- **True globes**: 9, 22, 35, 48
- **Enemy starts (from your view)**: 14, 27, 40

**Total safe positions**: 1, 9, 14, 22, 27, 35, 40, 48 (8 positions)

## Implementation Impact

Our visualization correctly shows:
- Index 0-57 (58 total positions)
- Stars at the correct positions
- Globes at positions 9, 22, 35, 48
- Start at position 1 (with globe icon, which is accurate for visualization)

The visualization is **CORRECT** according to ludopy's system!

