#!/usr/bin/env python3
"""
Test script for board visualization with track indices.
"""

from rl_agent_ludo.utils.visualizer import visualize_board_indices

# Test 1: Board with track indices (0-56)
print("Creating board with track indices...")
visualize_board_indices(
    save_path="src/rl_agent_ludo/assests_outputs/board_with_indices.png",
    show_track_indices=True,
    show_tile_coords=False,
    show_legend=False
)

# Test 2: Board with track indices and legend
print("Creating board with track indices and legend...")
visualize_board_indices(
    save_path="src/rl_agent_ludo/assests_outputs/board_with_indices_and_legend.png",
    show_track_indices=True,
    show_tile_coords=False,
    show_legend=True
)

# Test 3: Board with both track indices and tile coordinates
print("Creating board with track indices and tile coordinates...")
visualize_board_indices(
    save_path="src/rl_agent_ludo/assests_outputs/board_with_both.png",
    show_track_indices=True,
    show_tile_coords=True,
    show_legend=False
)

# Test 4: Board with only tile coordinates (original behavior)
print("Creating board with only tile coordinates...")
visualize_board_indices(
    save_path="src/rl_agent_ludo/assests_outputs/board_tile_coords_only.png",
    show_track_indices=False,
    show_tile_coords=True,
    show_legend=False
)

print("\n✅ All test visualizations created successfully!")
print("Check the src/rl_agent_ludo/assests_outputs/ folder for the images.")

