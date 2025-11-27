# Ludo Board Index Quick Reference

## Visual Reference

Open `ludo_board_annotated.png` to see the complete board with all indices labeled.

## Index Mapping (Player 0 Perspective)

### Main Track (0-50)

The main track goes around the board in this order:

```
Index  | Grid [row, col] | Type        | Notes
-------|-----------------|-------------|---------------------------
0      | [14, 7]         | START       | Globe (safe), Player 0 start
1-4    | ...             | NORMAL      | 
5      | [9, 6]          | STAR        | Jump to next star
6-7    | ...             | NORMAL      |
8      | [9, 3]          | GLOBE       | Safe position
9-10   | ...             | NORMAL      |
11     | [8, 1]          | STAR        | Jump to next star
12     | [7, 1]          | NORMAL      |
13     | [7, 2]          | START       | Player 1 (Yellow) start, Globe
14-16  | ...             | NORMAL      |
17     | [7, 6]          | STAR        | Jump to next star
18-20  | ...             | NORMAL      |
21     | [3, 7]          | GLOBE       | Safe position
22-23  | ...             | NORMAL      |
24     | [1, 8]          | STAR        | Jump to next star
25     | [1, 9]          | NORMAL      |
26     | [2, 9]          | START       | Player 2 (Blue) start, Globe
27-29  | ...             | NORMAL      |
30     | [6, 9]          | STAR        | Jump to next star
31-33  | ...             | NORMAL      |
34     | [7, 13]         | GLOBE       | Safe position
35-36  | ...             | NORMAL      |
37     | [8, 15]         | STAR        | Jump to next star
38     | [9, 15]         | NORMAL      |
39     | [9, 14]         | START       | Player 3 (Pink) start, Globe
40-42  | ...             | NORMAL      |
43     | [9, 10]         | STAR        | Jump to next star
44-46  | ...             | NORMAL      |
47     | [13, 9]         | GLOBE       | Safe (CHOKE POINT)
48-49  | ...             | NORMAL      |
50     | [15, 8]         | STAR        | Jump to next star (or goal!)
```

### Home Stretch (51-55)

Private safe zone for Player 0:

```
Index  | Grid [row, col] | Notes
-------|-----------------|---------------------------
51     | [14, 8]         | First home stretch tile
52     | [13, 8]         | 
53     | [12, 8]         | 
54     | [11, 8]         | 
55     | [10, 8]         | Last tile before goal
```

### Goal (56)

```
Index  | Grid [row, col] | Notes
-------|-----------------|---------------------------
56     | [8, 8]          | Center - WIN CONDITION
```

## Special Position Summary

### Stars (8 total)
Indices: **5, 11, 17, 24, 30, 37, 43, 50**

Landing on a star makes you jump to the next star:
- 5 → 11 → 17 → 24 → 30 → 37 → 43 → 50 → (Goal if exact)

### Globes (5 total on main track)
Indices: **0, 8, 21, 34, 47**

Safe positions where pieces cannot be captured. Multiple pieces can occupy a globe.

### Player Start Positions
- **Player 0 (Green)**: Index 0
- **Player 1 (Yellow)**: Index 13  
- **Player 2 (Blue)**: Index 26
- **Player 3 (Pink)**: Index 39

All start positions are also globes (safe).

### Most Strategic Positions

1. **Index 47** - Choke point, last safe globe before home
2. **Index 50** - Last star, can jump to goal
3. **Index 51** - Entrance to home stretch
4. **Index 0** - Start position (safe)

## Movement Rules

1. **From Home Yard**: Roll a 6 to move to Index 0
2. **Main Track**: Move clockwise around board (0 → 1 → 2 → ... → 50)
3. **Home Stretch Entry**: After Index 50, move to Index 51 (home stretch)
4. **Goal Entry**: Must land exactly on Index 56
5. **Star Jumps**: Landing on star = automatic jump to next star
6. **Globes**: Safe from capture, multiple pieces allowed

## Grid Coordinate System

Ludopy uses [row, col] coordinates:
- **Rows**: 0-16 (top to bottom)
- **Columns**: 0-19 (left to right)
- **Origin**: [0, 0] is top-left corner
- **Center**: [8, 8] is the goal area

## Color Coding in Visualization

| Color | Meaning |
|-------|---------|
| **Green** | Player 0 (home yard & stretch) |
| **Brown** | Main track (shared) |
| **Cyan** | Goal area |
| **Yellow** | Player 1 areas |
| **Orange** | Player 2 areas |
| **Pink** | Player 3 areas |

## Quick Commands

```bash
# Generate board visualization
python visualize_board.py

# Run tests
python test_visualizer.py

# Custom size
python -c "from src.rl_agent_ludo.utils.visualizer import visualize_board_indices; visualize_board_indices('custom.png', width=1920)"
```

## Notes

- **Player Perspective**: This reference shows Player 0's view (starting at bottom-right)
- **Other Players**: Each player has their own perspective with the same 57-position track
- **Coordinate Mapping**: Grid positions are from ludopy's exact coordinate system
- **Index Accuracy**: All positions verified against ludopy's TRACK_PLAYER_1 definition

## Visual Files

- `ludo_board_annotated.png` - Main reference (with legend)
- `test_no_legend.png` - Clean board (indices only)
- `test_default.png` - Default size test
- `test_1920w.png` - High resolution version
- `test_800h.png` - Compact version

For complete documentation, see `VISUALIZATION_README.md`.

