# Standard Board Visualizer

## Overview

The `StandardLudoBoardVisualizer` provides a **proper cross-shaped Ludo board** layout matching the classic game design, replacing the linear/horizontal track visualization.

## Key Features

### ✅ Classic Ludo Board Layout
- **17 rows × 20 columns** tile grid (64×64 pixels per tile)
- Cross-shaped track with proper player home areas
- Central goal area (gold-colored 3×3 center)
- Four corner home zones for each player

### ✅ Standard Game Elements
- **Globe tiles** (safe zones where pieces cannot be captured)
- **Star tiles** (bonus positions, also safe)
- **Player-specific goal paths** leading to center
- **Home areas** in each corner for starting positions

### ✅ Visual Improvements
- Matching colors from original Ludo game
- Proper piece collision handling (multiple pieces on same tile)
- Legend showing player colors and game elements
- Clean header panel with game information
- Winner announcement banner

## File Structure

```
src/rl_agent_ludo/environment/
├── standard_board_visualizer.py  # NEW: Standard board implementation
├── enhanced_visualizer.py         # Original enhanced visualizer
└── curriculum_visualizer.py       # Original curriculum visualizer

experiments/
├── demo_visual.py                 # UPDATED: Now uses standard board
└── test_standard_board.py         # NEW: Test suite for visualizer
```

## Usage

### Basic Usage

```python
from rl_agent_ludo.environment.standard_board_visualizer import StandardLudoBoardVisualizer

# Create visualizer
viz = StandardLudoBoardVisualizer(
    level=5,              # Curriculum level
    num_players=4,        # Number of players (2 or 4)
    tokens_per_player=2   # Tokens per player (1 or 2)
)

# Render game state
state_info = {
    'player_positions': [[1, 15], [5, 20], [10, 25], [8, 30]],  # Token positions
    'current_player': 0,   # Current player index
    'dice': 6,             # Dice value
    'step': 42,            # Step count
    'winner': None         # Winner (or None if ongoing)
}

viz.render(state_info)
viz.close()
```

### Run Demo with Standard Board

```bash
# Run level 5 demo with standard board
python experiments/demo_visual.py --level 5 --episodes 3

# Test the visualizer
python experiments/test_standard_board.py
```

## Board Coordinates

### Position Mapping
- **Position 0**: Home (starting area)
- **Positions 1-57**: Track positions (clockwise around board)
- **Positions 58-60**: Goal area (center)

### Board Layout
```
     [TOP-LEFT]      [TOP-CENTER]      [TOP-RIGHT]
         P1            TRACK P1            P2
       HOME          ← ← ← ← ← ←          HOME
                    TRACK P3              
                       ↓
                    
[LEFT]  TRACK P2 →  [CENTER]  ← TRACK P4  [RIGHT]
                      GOAL
                       ↑
                    
                    TRACK P4
       HOME         → → → → → →         HOME
         P3          TRACK P2              P4
   [BOTTOM-LEFT]   [BOTTOM-CENTER]   [BOTTOM-RIGHT]
```

### Player Starting Positions (Tile Coordinates)
- **Player 0** (Green): Bottom-left, starts at [14, 7]
- **Player 1** (Yellow): Top-left, starts at [7, 2]
- **Player 2** (Blue): Top-right, starts at [2, 9]
- **Player 3** (Red): Bottom-right, starts at [9, 14]

### Safe Zones (Globe Tiles)
```python
GLOBE_TILES = [
    [14, 7], [7, 2], [2, 9], [9, 14],  # Starting positions
    [13, 9], [9, 3], [3, 7], [7, 13]   # Middle safe zones
]
```

### Star Tiles (Bonus Positions)
```python
STAR_TILES = [
    [15, 8], [8, 1], [1, 8], [8, 15],  # Outer stars
    [10, 7], [7, 6], [6, 9], [9, 10]   # Inner stars
]
```

## Color Scheme

### Player Colors (BGR format)
```python
PLAYER_COLORS = {
    0: (39, 214, 74),    # Green (Agent)
    1: (105, 219, 210),  # Yellow
    2: (46, 171, 255),   # Blue
    3: (255, 107, 107),  # Red
}
```

### Board Colors
- **Background**: (186, 202, 215) - Light blue-gray
- **Track**: (174, 95, 74) - Brown
- **Goal**: (234, 255, 0) - Gold
- **Text**: (0, 0, 0) - Black

## Features Comparison

| Feature | Enhanced Visualizer | Standard Board Visualizer |
|---------|-------------------|-------------------------|
| Board Layout | Horizontal tracks | Cross-shaped (classic) |
| Tile Grid | Variable | 17×20 fixed grid |
| Home Areas | Simple boxes | Proper corner zones |
| Goal Area | Rectangle | 3×3 center (gold) |
| Safe Zones | Color-coded | Globes with icons |
| Star Tiles | Simple markers | Star icons |
| Legend | Right side | Bottom-right overlay |
| Asset Support | Limited | Glob/star images |

## Advanced Features

### 1. Piece Collision Handling
Automatically offsets multiple pieces on the same tile:
```python
# Single piece: centered
# 2 pieces: diagonal offsets
# 3-4 pieces: all four corners
```

### 2. Legend System
- Player color indicators with labels
- Safe zone (globe) explanation
- Star tile explanation
- Matching font style (FONT_HERSHEY_DUPLEX)

### 3. Game State Display
- Step counter
- Current player (color-coded)
- Dice value
- Winner banner (when game ends)

## Testing

Run the test suite:
```bash
python experiments/test_standard_board.py
```

Tests include:
1. **Empty board** - All pieces at home
2. **Game progression** - Start → Mid → End → Winner
3. **Collision handling** - Multiple pieces on same tile

## Integration with Curriculum Levels

The visualizer works seamlessly with all curriculum levels:

```python
# Level 1-2: 2 players, 1 token each
viz = StandardLudoBoardVisualizer(level=1, num_players=2, tokens_per_player=1)

# Level 3-4: 2 players, 2 tokens each
viz = StandardLudoBoardVisualizer(level=3, num_players=2, tokens_per_player=2)

# Level 5: 4 players, 2 tokens each
viz = StandardLudoBoardVisualizer(level=5, num_players=4, tokens_per_player=2)
```

## Assets

The visualizer uses optional image assets from `src/rl_agent_ludo/ludo/assets/`:
- `glob.png` - Globe/safe zone icon (40×40 pixels)
- `star.png` - Star/bonus icon (40×40 pixels)

If assets are not found, fallback shapes are drawn.

## Performance

- **Board Size**: 1280×1088 pixels (20 cols × 17 rows × 64px)
- **Render Speed**: ~60 FPS (single frame)
- **Memory**: Minimal (single board buffer)

## Known Limitations

1. **Position Mapping**: Currently assumes linear track positions (0-60)
   - Works for curriculum levels 1-5
   - May need adjustment for full Ludo rules (99 positions)

2. **Home Area**: All tokens at home (position 0) appear at first home tile
   - Aesthetic limitation, doesn't affect gameplay

3. **Asset Dependency**: Requires glob/star images for best visuals
   - Gracefully degrades to simple shapes

## Future Enhancements

- [ ] Add dice visual (dots on white square)
- [ ] Animate piece movements
- [ ] Highlight valid moves
- [ ] Add player turn indicator
- [ ] Support custom board themes
- [ ] Add move history replay

## Troubleshooting

### Issue: Board not displaying
**Solution**: Check if OpenCV is installed:
```bash
pip install opencv-python
```

### Issue: Assets not loading
**Solution**: Verify asset path:
```bash
ls src/rl_agent_ludo/ludo/assets/
```

### Issue: Pieces overlapping incorrectly
**Solution**: Check position values are in range 0-60

## Credits

Based on the original Ludo visualizer from:
- `src/rl_agent_ludo/ludo/visualizer.py`
- Standard Ludo board layout and tile coordinates
- Globe and star asset integration

## License

Same as project license.

