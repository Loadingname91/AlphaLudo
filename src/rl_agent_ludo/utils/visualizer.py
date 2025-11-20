"""
Custom visualizer for Ludo board that extends ludopy's visualization capabilities.
Imports and uses ludopy's visualizer module for all drawing operations.
"""

import numpy as np
import cv2
from pathlib import Path
from ludopy import visualizer as ludopy_vis

# Re-export commonly used constants from ludopy
TRACKS = ludopy_vis.TRACKS
HOME_TAILES = ludopy_vis.HOME_TAILES
PLAYER_COLORS = ludopy_vis.PLAYER_COLORS
BOARD_TAILE_SIZE = ludopy_vis.BOARD_TAILE_SIZE
GOAL_COLOR = ludopy_vis.GOAL_COLOR


class BoardVisualizer:
    """
    Visualizes the Ludo board with annotated indices and special positions.
    
    Features:
    - Uses ludopy's exact coordinate system
    - Shows all board indices (0-56 + goal)
    - Responsive sizing based on screen resolution
    - Color-coded special positions
    """
    
    @staticmethod
    def draw_basic_board(draw_taile_number=False):
        """
        Draw the basic Ludo board using ludopy's visualizer.
        
        Args:
            draw_taile_number: If True, show tile coordinates
            
        Returns:
            numpy array representing the board image
        """
        return ludopy_vis.draw_basic_board(draw_taile_number=draw_taile_number)
    
    @staticmethod
    def draw_board_with_state(pieces, dice=-1, current_player=0, round_number=0):
        """
        Draw the board with current game state.
        
        Args:
            pieces: Array of player pieces positions
            dice: Current dice value (-1 for none)
            current_player: Current player index (0-3)
            round_number: Current round number
            
        Returns:
            numpy array representing the board image
        """
        return ludopy_vis.make_img_of_board(pieces, dice, current_player, round_number)
    
    @staticmethod
    def save_video(filename, history, fps=8, frame_size=None, fourcc=None):
        """
        Save game history as a video file.
        
        Args:
            filename: Output video filename (e.g., 'game.mp4' or 'game.avi')
            history: Game history from ludopy
            fps: Frames per second
            frame_size: Optional frame size tuple (width, height)
            fourcc: Optional fourcc codec
        """
        ludopy_vis.save_hist_video(filename, history, fps=fps, frame_size=frame_size, fourcc=fourcc)
    
    @staticmethod
    def get_tile_position(player, piece_position):
        """
        Get the tile coordinates for a piece position.
        
        Args:
            player: Player index (0-3)
            piece_position: Position on track (0 = home, 1-57 = on track, 59 = goal)
            
        Returns:
            Tuple of (n, m) tile coordinates, or None if invalid
        """
        if piece_position == 0:
            # Piece is at home - return first home tile
            return tuple(HOME_TAILES[player][0])
        elif 1 <= piece_position <= 57:
            # Piece is on the track
            track = TRACKS[player]
            if piece_position - 1 < len(track):
                return tuple(track[piece_position - 1])
        elif piece_position == 59:
            # Piece is in goal
            return (8, 8)  # Center of goal area
        return None



def visualize_board_indices(save_path, show=False, show_track_indices=True, show_tile_coords=False, show_legend=False):
    """
    Visualize the board with annotated indices and special positions.
    
    Args:
        save_path: Path to save the board image (e.g., 'board.png')
        show: If True, display the image in a window
        show_track_indices: If True, show game board indices (0-56 for track positions)
        show_tile_coords: If True, show tile coordinates (n,m format)
        show_legend: If True, add a legend explaining special positions
        
    Returns:
        numpy array of the board image
    """
    # Draw the basic board with or without tile coordinates
    board = BoardVisualizer.draw_basic_board(draw_taile_number=show_tile_coords)
    
    # Add track indices if requested
    if show_track_indices:
        board = _draw_track_indices(board)
    
    # Add legend if requested
    if show_legend:
        board = _add_legend_to_board(board)
    
    # Create directory if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert from RGB to BGR for OpenCV
    board_bgr = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(str(save_path), board_bgr)
    print(f"Board visualization saved to: {save_path}")
    
    # Optionally display the image
    if show:
        cv2.imshow('Ludo Board', board_bgr)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return board


def _draw_track_indices(board):
    """
    Draw the game board indices on the board.
    - Index 0: Home (off board) - marked on home tiles with "H"
    - Indices 1-51: Common track (starting at glob position)
    - Indices 52-56: Goal area (player-specific)
    - Index 57+: Goal (center)
    
    Args:
        board: numpy array of the board image
        
    Returns:
        numpy array of the board with track indices drawn
    """
    # Draw home markers (index 0) for each player on their home tiles
    for player in range(4):
        home_tiles = HOME_TAILES[player]
        for tile_idx, (n, m) in enumerate(home_tiles):
            center = ludopy_vis.get_taile_cord(n, m)[4]
            
            # Draw colored circle with "H" for home
            cv2.circle(board, center, 15, (255, 255, 255), -1)
            cv2.circle(board, center, 15, PLAYER_COLORS[player], 2)
            ludopy_vis.draw_text(board, "H", center, PLAYER_COLORS[player], 
                                thickness=2, fontScale=0.5)
    
    # Draw common track (indices 1-51, starting at glob position)
    # Index 1 = TRACKS[0][0] (first position on track, the starting glob)
    for i in range(51):  # This covers positions 1-51 on the track
        n, m = TRACKS[0][i]
        center = ludopy_vis.get_taile_cord(n, m)[4]
        
        # Draw white background circle for better visibility
        cv2.circle(board, center, 15, (255, 255, 255), -1)
        cv2.circle(board, center, 15, (0, 0, 0), 1)
        
        # Draw the index number (i+1 because track starts at 1, not 0)
        ludopy_vis.draw_text(board, str(i + 1), center, (0, 0, 0), 
                            thickness=2, fontScale=0.5)
    
    # Draw goal areas for each player (indices 52-57 in their respective goal areas)
    for player in range(4):
        track = TRACKS[player]
        # Draw indices 52-57 (goal area for each player)
        # track[51] = index 52, track[52] = index 53, ..., track[56] = index 57
        for i in range(51, min(57, len(track))):
            n, m = track[i]
            center = ludopy_vis.get_taile_cord(n, m)[4]
            
            # Draw colored background circle matching player color
            cv2.circle(board, center, 15, (255, 255, 255), -1)
            cv2.circle(board, center, 15, PLAYER_COLORS[player], 2)
            
            # Draw the index number (i+1 because indices are offset by 1)
            ludopy_vis.draw_text(board, str(i + 1), center, PLAYER_COLORS[player], 
                                thickness=2, fontScale=0.5)
    
    # Draw goal center (index 59/goal - pieces at position 59 have reached goal)
    goal_center = ludopy_vis.get_taile_cord(8, 8)[4]
    cv2.circle(board, goal_center, 20, (255, 255, 255), -1)
    cv2.circle(board, goal_center, 20, (0, 0, 0), 2)
    ludopy_vis.draw_text(board, "GOAL", goal_center, (0, 0, 0), 
                        thickness=2, fontScale=0.4)
    
    return board


def _add_legend_to_board(board):
    """
    Add a legend to the board explaining special positions.
    
    Args:
        board: numpy array of the board image
        
    Returns:
        numpy array of the board with legend added
    """
    # Get current board dimensions
    height, width = board.shape[:2]
    
    # Create legend area (white background)
    legend_height = 240
    legend_width = width
    legend = np.full((legend_height, legend_width, 3), 255, dtype=np.uint8)
    
    # Define legend items
    legend_items = [
        ("Home (0): H marker", (200, 200, 200), ""),
        ("Track: 1-51 (start at glob)", ludopy_vis.TAILE_BACKGROUND_COLOR, ""),
        ("Goal Areas: 52-57", ludopy_vis.PLAYER_1_AREAL_COLOR, ""),
        ("Safe Zones (Stars)", ludopy_vis.TAILE_BACKGROUND_COLOR, "⭐"),
        ("Globe Positions", ludopy_vis.TAILE_BACKGROUND_COLOR, "🌐"),
        ("Player 1 (Green)", ludopy_vis.PLAYER_1_COLOR, ""),
        ("Player 2 (Yellow)", ludopy_vis.PLAYER_2_COLOR, ""),
        ("Player 3 (Blue)", ludopy_vis.PLAYER_3_COLOR, ""),
        ("Player 4 (Red)", ludopy_vis.PLAYER_4_COLOR, ""),
    ]
    
    # Draw legend title
    ludopy_vis.draw_text(legend, "Legend", (width // 2, 30), (0, 0, 0), 
                        thickness=2, fontScale=1.2)
    
    # Draw legend items in two columns
    y_start = 70
    y_spacing = 35
    col1_x = 100
    col2_x = width // 2 + 100
    
    for i, (label, color, symbol) in enumerate(legend_items):
        # Determine column position (5 items per column)
        if i < 5:
            x_pos = col1_x
            y_pos = y_start + (i * y_spacing)
        else:
            x_pos = col2_x
            y_pos = y_start + ((i - 5) * y_spacing)
        
        # Draw color box
        box_size = 20
        cv2.rectangle(legend, 
                     (x_pos - 30, y_pos - 10), 
                     (x_pos - 30 + box_size, y_pos - 10 + box_size), 
                     color, -1)
        cv2.rectangle(legend, 
                     (x_pos - 30, y_pos - 10), 
                     (x_pos - 30 + box_size, y_pos - 10 + box_size), 
                     (0, 0, 0), 1)
        
        # Draw label
        ludopy_vis.draw_text(legend, label, (x_pos, y_pos), (0, 0, 0), 
                            thickness=1, fontScale=0.6)
    
    # Concatenate board and legend vertically
    combined = np.vstack([board, legend])
    
    return combined