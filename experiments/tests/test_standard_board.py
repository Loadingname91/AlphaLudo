"""
Test script for the standard board visualizer.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.standard_board_visualizer import StandardLudoBoardVisualizer
import cv2

def test_empty_board():
    """Test rendering an empty board."""
    print("Testing empty board...")
    
    viz = StandardLudoBoardVisualizer(level=5, num_players=4, tokens_per_player=2)
    
    state_info = {
        'player_positions': [
            [0, 0],  # Player 0: both tokens at home
            [0, 0],  # Player 1: both tokens at home
            [0, 0],  # Player 2: both tokens at home
            [0, 0],  # Player 3: both tokens at home
        ],
        'current_player': 0,
        'dice': 6,
        'step': 0
    }
    
    viz.render(state_info)
    print("Empty board rendered. Press any key in the window to continue...")
    cv2.waitKey(0)
    viz.close()

def test_game_progression():
    """Test rendering a game in progress."""
    print("\nTesting game progression...")
    
    viz = StandardLudoBoardVisualizer(level=5, num_players=4, tokens_per_player=2)
    
    # Simulate a game progression
    positions = [
        # Start
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        # Early game
        [[1, 5], [1, 3], [1, 8], [1, 2]],
        # Mid game
        [[15, 22], [20, 18], [25, 30], [12, 28]],
        # Late game
        [[45, 50], [40, 55], [48, 52], [42, 58]],
        # Winner
        [[60, 60], [55, 58], [52, 54], [48, 50]],
    ]
    
    for i, pos in enumerate(positions):
        state_info = {
            'player_positions': pos,
            'current_player': i % 4,
            'dice': (i % 6) + 1,
            'step': i * 10,
            'winner': 0 if i == len(positions) - 1 else None
        }
        
        viz.render(state_info)
        print(f"Step {i+1}/{len(positions)}: {'WINNER!' if state_info['winner'] is not None else 'In progress'}")
        time.sleep(1.5)
    
    print("Game progression complete. Press any key to close...")
    cv2.waitKey(0)
    viz.close()

def test_collision_handling():
    """Test multiple pieces on same tile."""
    print("\nTesting piece collision handling...")
    
    viz = StandardLudoBoardVisualizer(level=5, num_players=4, tokens_per_player=2)
    
    # Put multiple pieces on same positions
    state_info = {
        'player_positions': [
            [10, 10],  # Both tokens of player 0 on position 10
            [10, 20],  # Player 1: one at 10 (collision!), one at 20
            [25, 25],  # Player 2: both at 25
            [25, 30],  # Player 3: one at 25 (collision!), one at 30
        ],
        'current_player': 0,
        'dice': 3,
        'step': 42
    }
    
    viz.render(state_info)
    print("Collision handling rendered. Press any key to continue...")
    cv2.waitKey(0)
    viz.close()

def main():
    print("="*60)
    print("Standard Board Visualizer Test Suite")
    print("="*60)
    
    try:
        test_empty_board()
        test_game_progression()
        test_collision_handling()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nNow you can run: python experiments/demo_visual.py --level 5 --episodes 3")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

