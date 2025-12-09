# Visualization Guide - Curriculum Ludo

## 🎮 Commands to Watch Agents Play (with CV2 Graphics)

> **Note**: For detailed information about the standard Ludo board visualizer, see [docs/STANDARD_BOARD_VISUALIZER.md](docs/STANDARD_BOARD_VISUALIZER.md)

### Quick Start

```bash
# Watch Level 1 - Basic Movement (simple track)
python experiments/demo_visual.py --level 1 --episodes 3

# Watch Level 2 - With Captures
python experiments/demo_visual.py --level 2 --episodes 3

# Watch Level 3 - Multi-Token Strategy
python experiments/demo_visual.py --level 3 --episodes 3

# Watch Level 4 - Stochastic (with dice!)
python experiments/demo_visual.py --level 4 --episodes 3

# Watch Level 5 - 4-Player Chaos!
python experiments/demo_visual.py --level 5 --episodes 2
```

### Controls
- **Watch**: CV2 window pops up showing the board
- **Press 'q'**: Quit early
- **Close window**: Stop visualization

---

## 🖼️ What You'll See

### Level 1-2: Simple Linear Track
- **Green track**: The game board (0 to 60 positions)
- **Gold area**: Goal zone
- **Green circle (P0)**: Agent's token
- **Red circle (P1)**: Opponent's token
- Position markers every 10 steps
- Real-time movement as the game progresses

### Level 3-5: Multi-Token Board
- **Separate tracks** for each player
- **Multiple tokens** per player shown as numbered circles
- **Home position**: Left side (position -1)
- **Goal position**: Gold zone on right
- Dice visualization (for Level 4-5)
- Current player highlighted

### Game Info Display
- Step counter
- Current player turn
- Dice value (Levels 4-5)
- Visual dice representation
- Winner announcement when game ends

---

## 🐧 Running on WSL/Linux Without Display

If you're on WSL or Linux without a display, you'll need X11 forwarding:

### Option 1: X11 Forwarding (WSL2)

1. Install X Server on Windows:
   - Download **VcXsrv** or **Xming**
   - Run XLaunch with "Disable access control" checked

2. Set DISPLAY environment variable:
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

3. Test with:
```bash
python -c "import cv2; cv2.imshow('test', [[255]]); cv2.waitKey(1000)"
```

4. If successful, run the demo scripts!

### Option 2: Save Visualization to Video

If you can't get X11 working, use the recording mode:

```bash
# Record Level 5 gameplay to video file
python experiments/demo_visual.py --level 5 --episodes 3 --save video_level5.mp4
```

*Note: Video saving feature coming soon if needed!*

### Option 3: Use Native Windows Python

If on WSL, install Python on Windows and run from Windows terminal:
```bash
# From Windows PowerShell/CMD (not WSL)
cd C:\path\to\RLagentLudo
.venv\Scripts\activate  # Windows venv
python experiments\demo_visual.py --level 1 --episodes 3
```

---

## 📊 Alternative: Static Visualizations

Already created! Check these out:
```bash
# View the dashboard and plots we generated
ls results/visualizations/

# Files available:
# - dashboard.png          (comprehensive overview)
# - win_rates.png          (level comparison)
# - rewards.png            (reward analysis)
# - episode_lengths.png    (efficiency metrics)
```

Open these PNG files to see the training results!

---

## 🔧 Troubleshooting

### "cannot connect to X server"
- X11 forwarding not set up (see Option 1 above)
- Or run on Windows Python directly (Option 3)

### Window appears but is blank/frozen
- Update OpenCV: `pip install --upgrade opencv-python`
- Try: `export QT_QPA_PLATFORM=offscreen`

### "module 'cv2' has no attribute..."
- OpenCV version issue
- Update: `pip install opencv-python>=4.8.0`

---

## 🎯 What Makes This Cool

1. **Real-time gameplay**: Watch the agent make decisions live
2. **Color-coded players**: Easy to track who's who
3. **Visual feedback**: See captures, movements, dice rolls
4. **Multiple levels**: Different complexity visualized differently
5. **Educational**: Perfect for showing your professor!

---

## 📝 Technical Details

### Visualization Features
- **1200x800 pixel** window
- **100ms refresh rate** (adjustable)
- **Color scheme**:
  - Green (Player 0 - Agent)
  - Red (Player 1)
  - Blue (Player 2)
  - Yellow (Player 3)
  - Gold (Goal zones)

### Files Involved
- `src/rl_agent_ludo/environment/curriculum_visualizer.py` - Core visualizer
- `experiments/demo_visual.py` - Demo script with agent
- Level environments (level1_simple.py, etc.) - Game logic

---

## 🚀 Next Steps

1. **Try Level 1 first**: Simplest visualization
2. **Work up to Level 5**: Most impressive (4 players!)
3. **Record for presentation**: Take screenshots or screen recording
4. **Show your professor**: Visual proof of learning!

Enjoy watching your trained agents play! 🎮
