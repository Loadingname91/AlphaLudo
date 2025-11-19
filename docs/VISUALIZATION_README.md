# Ludo Board Visualization System

## Overview

The visualization system creates static, annotated board images showing all index positions (0-56 + goal) using **ludopy's exact coordinate system** for accurate positioning.

## Key Features

✅ **Accurate Positioning**: Uses ludopy's 17x20 grid system (64px tiles)
✅ **Responsive Sizing**: Auto-adjusts to screen resolution or custom dimensions  
✅ **Index Annotations**: Shows all board indices (0-56) on tiles
✅ **Color-Coded Positions**: Different colors for tracks, home yards, and goal
✅ **Legend Support**: Optional legend showing position types
✅ **Star & Globe Icons**: Uses ludopy's actual icon images

## Quick Start

### Basic Usage

```python
from src.rl_agent_ludo.utils.visualizer import visualize_board_indices

# Simple: auto-size based on screen, save and show
visualize_board_indices(
    save_path="board.png",
    show=True
)
```

### Custom Sizing

```python
# Specify width (height auto-calculated)
visualize_board_indices(
    save_path="board_1920.png",
    width=1920,
    show=False
)

# Specify height (width auto-calculated)
visualize_board_indices(
    save_path="board_800.png",
    height=800,
    show=False
)

# Specify both
visualize_board_indices(
    save_path="board_custom.png",
    width=1600,
    height=900,
    show=True
)
```

### Advanced: Using BoardVisualizer Class

```python
from src.rl_agent_ludo.utils.visualizer import BoardVisualizer

# Create visualizer with custom size
visualizer = BoardVisualizer(target_width=1920)

# Save with options
visualizer.save(
    "board.png",
    show_legend=True,   # Include legend
    show_indices=True   # Show index numbers
)

# Display in window
visualizer.show(wait_key=0)  # 0 = wait for key press

# Cleanup
visualizer.close()
```

## Board Index System

The visualization shows indices from Player 0's perspective (bottom-right starting position):

### Index Ranges

| Range | Description | Color |
|-------|-------------|-------|
| **0** | Home yard position for Player 0 | Green |
| **0-50** | Main track (shared by all players) | Brown |
| **51-55** | Home stretch (Player 0 only) | Green |
| **56** | Goal (center) | Cyan |

### Special Positions

- **Stars** (5, 11, 17, 24, 30, 37, 43, 50): Jump to next star
- **Globes** (0, 8, 21, 34, 47): Safe positions (cannot be captured)
- **Player Starts**: 0 (Green), 13 (Yellow), 26 (Blue), 39 (Pink)

## Technical Details

### Coordinate System

Uses ludopy's grid-based system:
- Board: 17 rows × 20 columns
- Tile size: 64×64 pixels
- Base board size: 1088×1280 pixels (before scaling)
- Grid coordinates: [row, col] where row=0 is top, col=0 is left

### Track Definition

The main track (TRACK_PLAYER_0) defines 57 grid positions:
- Indices 0-50: Main track around the board
- Indices 51-55: Home stretch leading to goal
- Index 56: Goal (center at grid position [8, 8])

Example positions:
```python
Index 0:  [14, 7]  # Start position (Globe)
Index 5:  [9, 6]   # First star
Index 51: [14, 8]  # Start of home stretch
Index 56: [8, 8]   # Goal (center)
```

### Color Scheme

| Element | Color (BGR) | Purpose |
|---------|-------------|---------|
| Background | (215, 202, 186) | Light beige |
| Track | (74, 95, 174) | Brown tiles |
| Player 0 | (39, 214, 74) | Green (home yard & stretch) |
| Player 1 | (105, 219, 210) | Yellow |
| Player 2 | (255, 171, 46) | Orange/Blue |
| Player 3 | (107, 107, 255) | Pink |
| Goal | (0, 255, 234) | Bright cyan |

## File Outputs

### Generated Files

- `ludo_board_annotated.png` - Main annotated board (default output)
- `test_default.png` - Default size test
- `test_1920w.png` - 1920px width test
- `test_800h.png` - 800px height test
- `test_no_legend.png` - No legend version

### File Sizes

- Default (1280×1088): ~170-180KB
- 1920px width: ~350-400KB
- 800px height: ~90-100KB

## Screen Adaptation

The visualizer automatically adapts to your screen resolution:

1. **Auto-detection**: Uses tkinter to get screen size
2. **Smart scaling**: Uses 80% of screen width/height
3. **Aspect ratio**: Maintains correct proportions (1088:1280)
4. **Fallback**: Uses 1280×1088 if detection fails

## Integration with LudoEnv

The BoardVisualizer can be used alongside LudoEnv for reference:

```python
from src.rl_agent_ludo.environment.ludo_env import LudoEnv
from src.rl_agent_ludo.utils.visualizer import visualize_board_indices

# Create static reference board
visualize_board_indices("reference_board.png", show=False)

# Use LudoEnv with real-time rendering
env = LudoEnv(render=True)
state = env.reset()

# During training, you can reference the static board
# to understand which index corresponds to which position
```

## Command Line Usage

```bash
# Generate annotated board
python visualize_board.py

# Run tests
python test_visualizer.py
```

## Requirements

- `opencv-python >= 4.8.0` - For rendering and image operations
- `numpy >= 1.24.0` - For array operations
- `ludopy >= 1.3.0` - For board structure and icons
- `tkinter` (optional) - For screen resolution detection

## Troubleshooting

### Issue: Index numbers not visible
**Solution**: Numbers are drawn with white text on black backgrounds at each tile center. Ensure `show_indices=True`.

### Issue: Icons not showing
**Solution**: Icons (stars/globes) require ludopy's PNG files. Check that ludopy is properly installed.

### Issue: Window too small/large
**Solution**: Specify custom width/height:
```python
visualize_board_indices(width=1600, height=1360)
```

### Issue: "tkinter not found" error
**Solution**: This is only a warning. The visualizer will use default dimensions (1280×1088).

## API Reference

### visualize_board_indices()

```python
def visualize_board_indices(
    save_path: Optional[str] = None,
    show: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None
```

**Parameters:**
- `save_path` - Path to save image (None = don't save)
- `show` - Display in window (True/False)
- `width` - Target width in pixels (None = auto)
- `height` - Target height in pixels (None = auto)

### BoardVisualizer Class

```python
class BoardVisualizer:
    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    )
    
    def render_annotated_board(
        self,
        show_legend: bool = True,
        show_indices: bool = True
    ) -> np.ndarray
    
    def save(
        self,
        filepath: str,
        show_legend: bool = True,
        show_indices: bool = True
    ) -> None
    
    def show(self, wait_key: int = 0) -> None
    
    def close(self) -> None
```

## Examples

### Example 1: Quick Reference Board

```python
from src.rl_agent_ludo.utils.visualizer import visualize_board_indices

visualize_board_indices("quick_ref.png", show=True)
```

### Example 2: High-Resolution for Printing

```python
from src.rl_agent_ludo.utils.visualizer import BoardVisualizer

vis = BoardVisualizer(target_width=3840)  # 4K width
vis.save("board_4k.png", show_legend=True, show_indices=True)
vis.close()
```

### Example 3: Multiple Versions

```python
from src.rl_agent_ludo.utils.visualizer import BoardVisualizer

# Full version with legend
vis = BoardVisualizer()
vis.save("board_full.png", show_legend=True, show_indices=True)

# Clean version without legend
vis.save("board_clean.png", show_legend=False, show_indices=True)

# Minimal version (no annotations)
vis.save("board_minimal.png", show_legend=False, show_indices=False)

vis.close()
```

## Future Enhancements

Potential improvements:
- [ ] Support for all 4 player perspectives
- [ ] Animated visualization showing piece movements
- [ ] Interactive mode with click-to-highlight
- [ ] Export to SVG for vector graphics
- [ ] Customizable color schemes
- [ ] Overlay current game state on reference board

## License

Part of the RLagentLudo project.

