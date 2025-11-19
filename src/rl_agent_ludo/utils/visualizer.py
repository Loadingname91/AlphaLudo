"""
Board Visualization Utility

Provides visualization of the Ludo board with annotated indices and special positions.
Uses ludopy's exact coordinate system for accurate positioning.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import os

# OpenCV imports (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    print("Warning: opencv-python (cv2) not installed. Visualization will be disabled.")

# Ludopy board constants (from ludopy.visualizer)
TILE_SIZE_FULL = np.array([64, 64])  # Each tile is 64x64 pixels
BOARD_TILE_SIZE = np.array([17, 20])  # Board is 17x20 tiles
BOARD_SIZE = TILE_SIZE_FULL * BOARD_TILE_SIZE  # Total board size: 1088x1280 pixels

# Define the track for Player 0 (Red - bottom right) using ludopy's coordinate system
# Format: [row, col] where row=0 is top, col=0 is left
#
# INDEXING SYSTEM:
# - Analyzer Index: Used in game logic (0 = off-board, 1-56 = on-board positions)
# - Visual Index: Shown on board visualization (0-55, since off-board is not displayed)
# - Mapping: Visual Index = Analyzer Index - 1


TRACK_PLAYER_0 = np.array([
    [-1, -1],  # 0: Off board (Home yard - not on track yet)
    [13, 7],   # 1: Globe, Start (Player 0 - Red - entry point after rolling 6)
    [12, 7],   # 2
    [11, 7],   # 3
    [10, 7],   # 4
    [9, 6],    # 5: Star
    [9, 5],    # 6
    [9, 4],    # 7
    [9, 3],    # 8
    [9, 2],    # 9: Globe
    [9, 1],    # 10
    [8, 1],    # 11
    [7, 1],    # 12: Star
    [7, 2],    # 13
    [7, 3],    # 14: Start (Player 1 - Green)
    [7, 4],    # 15
    [7, 5],    # 16
    [7, 6],    # 17
    [6, 7],    # 18: Star
    [5, 7],    # 19
    [4, 7],    # 20
    [3, 7],    # 21
    [2, 7],    # 22: Globe
    [1, 7],    # 23
    [1, 8],    # 24
    [1, 9],    # 25: Star
    [2, 9],    # 26
    [3, 9],    # 27: Start (Player 2 - Blue)
    [4, 9],    # 28
    [5, 9],    # 29
    [6, 9],    # 30
    [7, 10],   # 31: Star
    [7, 11],   # 32
    [7, 12],   # 33
    [7, 13],   # 34
    [7, 14],   # 35: Globe
    [7, 15],   # 36
    [8, 15],   # 37
    [9, 15],   # 38: Star
    [9, 14],   # 39
    [9, 13],   # 40: Start (Player 3 - Yellow)
    [9, 12],   # 41
    [9, 11],   # 42
    [9, 10],   # 43
    [10, 9],   # 44: Star
    [11, 9],   # 45
    [12, 9],   # 46
    [13, 9],   # 47
    [14, 9],   # 48: Globe
    [15, 9],   # 49
    [15, 8],   # 50
    # Home stretch (Player 0)
    [14, 8],   # 51: Star
    [13, 8],   # 52
    [12, 8],   # 53
    [11, 8],   # 54
    [10, 8],   # 55
    # Goal
    [8, 8],    # 56: Goal (center)
])

# Home yard positions for Player 0 (off-board starting positions)
HOME_YARD_PLAYER_0 = [
    [12, 3],   # Piece 0
    [14, 3],   # Piece 1
    [12, 5],   # Piece 2
    [14, 5]    # Piece 3
]


class BoardVisualizer:
    """
    Visualizes the Ludo board with annotated indices and special positions.
    
    Features:
    - Uses ludopy's exact coordinate system
    - Shows all board indices (0-56 + goal)
    - Responsive sizing based on screen resolution
    - Color-coded special positions
    """
    
    # Board feature definitions (indices for Player 0 perspective - analyzer indices)
    STARS = [5, 12, 18, 25, 31, 38, 44, 51]
    GLOBES = [1, 9, 22, 35, 48]
    PLAYER_STARTS = [1, 14, 27, 40]  # Start positions for each player
    HOME_STRETCH_START = 52
    GOAL = 56  # Goal is at analyzer index 56 (visual index 55)
    
    # Color definitions (BGR format for OpenCV)
    COLOR_BACKGROUND = (215, 202, 186)  # Light beige
    COLOR_TILE = (74, 95, 174)  # Brown
    COLOR_PLAYER_0 = (39, 214, 74)  # Green (Player 0/Red in ludopy)
    COLOR_PLAYER_1 = (105, 219, 210)  # Yellow (Player 1/Green in ludopy)
    COLOR_PLAYER_2 = (255, 171, 46)  # Orange (Player 2/Blue in ludopy)
    COLOR_PLAYER_3 = (107, 107, 255)  # Pink (Player 3/Yellow in ludopy)
    COLOR_GOAL = (0, 255, 234)  # Bright cyan
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    
    def __init__(self, target_width: Optional[int] = None, target_height: Optional[int] = None, show_indices: bool = True, show_legend: bool = True):
        """
        Initialize board visualizer.
        
        Args:
            target_width: Target width in pixels (None = use default 1280)
            target_height: Target height in pixels (None = use default 1088)
        """
        if not HAS_CV2:
            raise ImportError("opencv-python (cv2) is required for visualization")
        
        # Set target dimensions
        if target_width is None and target_height is None:
            # Default size
            self.target_width = BOARD_SIZE[1]  # 1280
            self.target_height = BOARD_SIZE[0]  # 1088
        elif target_width is not None and target_height is None:
            # Scale based on width, maintain aspect ratio
            self.target_width = target_width
            aspect_ratio = BOARD_SIZE[0] / BOARD_SIZE[1]
            self.target_height = int(target_width * aspect_ratio)
        elif target_height is not None and target_width is None:
            # Scale based on height, maintain aspect ratio
            self.target_height = target_height
            aspect_ratio = BOARD_SIZE[1] / BOARD_SIZE[0]
            self.target_width = int(target_height * aspect_ratio)
        else:
            # Both specified
            self.target_width = target_width
            self.target_height = target_height
        
        self.window_name = "Ludo Board - Index Reference"
        
        # Calculate scale factor
        self.scale_x = self.target_width / BOARD_SIZE[1]
        self.scale_y = self.target_height / BOARD_SIZE[0]

        self.show_indices = show_indices
        self.show_legend = show_legend
        # Load star and globe images if available
        self._load_icons()
    
    def _load_icons(self) -> None:
        """Load star and globe icons from ludopy package."""
        try:
            import ludopy
            ludopy_folder = os.path.dirname(ludopy.__file__)
            
            # Load glob icon
            glob_path = os.path.join(ludopy_folder, "glob.png")
            if os.path.exists(glob_path):
                self.glob_img = cv2.imread(glob_path)
                self.glob_img = cv2.cvtColor(self.glob_img, cv2.COLOR_BGR2RGB)
                self.glob_img = cv2.resize(self.glob_img, (40, 40))
                self.glob_mask = cv2.inRange(self.glob_img, (255, 255, 255), (255, 255, 255)) == 0
            else:
                self.glob_img = None
                self.glob_mask = None
            
            # Load star icon
            star_path = os.path.join(ludopy_folder, "star.png")
            if os.path.exists(star_path):
                self.star_img = cv2.imread(star_path)
                self.star_img = cv2.cvtColor(self.star_img, cv2.COLOR_BGR2RGB)
                self.star_img = cv2.resize(self.star_img, (40, 40))
                self.star_mask = cv2.inRange(self.star_img, (255, 255, 255), (255, 255, 255)) == 0
            else:
                self.star_img = None
                self.star_mask = None
        except:
            self.glob_img = None
            self.glob_mask = None
            self.star_img = None
            self.star_mask = None
    
    def _get_tile_center(self, row: int, col: int) -> Tuple[int, int]:
        """
        Get the center pixel coordinates for a tile at grid position (row, col).
        
        Args:
            row: Tile row (0-16)
            col: Tile column (0-19)
        
        Returns:
            (x, y) pixel coordinates of tile center
        """
        x = int((col + 0.5) * TILE_SIZE_FULL[0])
        y = int((row + 0.5) * TILE_SIZE_FULL[1])
        return (x, y)
    
    def _get_tile_bounds(self, row: int, col: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get top-left and bottom-right corners of a tile."""
        top_left = (col * TILE_SIZE_FULL[0], row * TILE_SIZE_FULL[1])
        bottom_right = ((col + 1) * TILE_SIZE_FULL[0], (row + 1) * TILE_SIZE_FULL[1])
        return top_left, bottom_right
    
    def _draw_tile(self, board: np.ndarray, row: int, col: int, 
                   fill_color: Tuple[int, int, int], 
                   border_color: Tuple[int, int, int] = None,
                   border_thickness: int = 2) -> None:
        """Draw a rectangular tile at the specified grid position."""
        top_left, bottom_right = self._get_tile_bounds(row, col)
        
        # Fill
        cv2.rectangle(board, top_left, bottom_right, fill_color, -1)
        
        # Border
        if border_color is not None:
            cv2.rectangle(board, top_left, bottom_right, border_color, border_thickness)
    
    def _draw_text_at_tile(self, board: np.ndarray, text: str, row: int, col: int,
                          color: Tuple[int, int, int], font_scale: float = 0.5,
                          thickness: int = 2, bg_color: Optional[Tuple[int, int, int]] = None) -> None:
        """Draw text centered at a tile position."""
        center = self._get_tile_center(row, col)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate bottom-left position for text (OpenCV uses bottom-left as origin for text)
        text_x = int(center[0] - text_width / 2)
        text_y = int(center[1] + text_height / 2)
        
        # Draw background rectangle if specified
        if bg_color is not None:
            padding = 4
            bg_top_left = (text_x - padding, text_y - text_height - padding)
            bg_bottom_right = (text_x + text_width + padding, text_y + padding)
            cv2.rectangle(board, bg_top_left, bg_bottom_right, bg_color, -1)
            cv2.rectangle(board, bg_top_left, bg_bottom_right, self.COLOR_WHITE, 1)
        
        # Draw text
        cv2.putText(board, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    def _put_image_at_tile(self, board: np.ndarray, image: np.ndarray, 
                          row: int, col: int, mask: Optional[np.ndarray] = None) -> None:
        """Place an image centered at a tile position."""
        center = self._get_tile_center(row, col)
        image_height, image_width = image.shape[:2]
        
        # Calculate top-left position
        y_start = center[1] - image_height // 2
        y_end = y_start + image_height
        x_start = center[0] - image_width // 2
        x_end = x_start + image_width
        
        # Ensure within bounds
        if y_start >= -1 and x_start >= -1 and y_end < board.shape[0] and x_end < board.shape[1]:
            if mask is not None:
                board[y_start:y_end, x_start:x_end][mask] = image[mask]
            else:
                board[y_start:y_end, x_start:x_end] = image
    
    def render_annotated_board(self, show_legend: bool = True, show_indices: bool = True) -> np.ndarray:
        """
        Render the complete board with annotations.

        Args:
            show_legend: Whether to add a legend
            show_indices: Whether to show index numbers on tiles
        
        Returns:
            Board image as numpy array (BGR format)
        """
        # Create blank board with background color
        board = np.full(shape=(*BOARD_SIZE, 3), fill_value=self.COLOR_BACKGROUND, dtype=np.uint8)
        
        # Draw goal area (center 3x3)
        for row in range(7, 10):
            for col in range(7, 10):
                self._draw_tile(board, row, col, self.COLOR_GOAL, self.COLOR_BLACK)
        
        # Draw home yards for each player
        self._draw_home_yards(board)
        
        # Draw main track tiles
        self._draw_track(board)
        
        # Draw home stretch for Player 0
        self._draw_home_stretch(board)
        
        # Overlay star and globe icons
        if self.star_img is not None:
            for idx in self.STARS:
                if idx < len(TRACK_PLAYER_0) and idx > 0:
                    row, col = TRACK_PLAYER_0[idx]
                    if row != -1 and col != -1:
                        self._put_image_at_tile(board, self.star_img, row, col, self.star_mask)
        
        if self.glob_img is not None:
            for idx in self.GLOBES:
                if idx < len(TRACK_PLAYER_0) and idx > 0:
                    row, col = TRACK_PLAYER_0[idx]
                    if row != -1 and col != -1:
                        self._put_image_at_tile(board, self.glob_img, row, col, self.glob_mask)
        
        # Draw index numbers
        if self.show_indices:
            self._draw_indices(board)
        
        # Resize to target dimensions
        board_resized = cv2.resize(board, (self.target_width, self.target_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Add title and legend
        if self.show_legend:
            board_resized = self._add_legend(board_resized)
        
        board_resized = self._add_title(board_resized)
        
        return board_resized
    
    def _draw_home_yards(self, board: np.ndarray) -> None:
        """Draw the home yard areas for all players."""
        # Player 0 (Green) - bottom right
        player_0_yards = [[12, 3], [14, 3], [12, 5], [14, 5]]
        for row, col in player_0_yards:
            self._draw_tile(board, row, col, self.COLOR_PLAYER_0, self.COLOR_BLACK)
        
        # Player 1 (Yellow) - top left
        player_1_yards = [[3, 2], [5, 2], [3, 4], [5, 4]]
        for row, col in player_1_yards:
            self._draw_tile(board, row, col, self.COLOR_PLAYER_1, self.COLOR_BLACK)
        
        # Player 2 (Blue) - top right
        player_2_yards = [[2, 11], [2, 13], [4, 11], [4, 13]]
        for row, col in player_2_yards:
            self._draw_tile(board, row, col, self.COLOR_PLAYER_2, self.COLOR_BLACK)
        
        # Player 3 (Pink) - bottom left
        player_3_yards = [[11, 12], [13, 12], [11, 14], [13, 14]]
        for row, col in player_3_yards:
            self._draw_tile(board, row, col, self.COLOR_PLAYER_3, self.COLOR_BLACK)
    
    def _draw_track(self, board: np.ndarray) -> None:
        """Draw the main track tiles (indices 1-50)."""
        for idx in range(1, 51):
            row, col = TRACK_PLAYER_0[idx]
            self._draw_tile(board, row, col, self.COLOR_TILE, self.COLOR_BLACK)
    
    def _draw_home_stretch(self, board: np.ndarray) -> None:
        """Draw the home stretch tiles for Player 0 (indices 51-55)."""
        for idx in range(51, 56):
            row, col = TRACK_PLAYER_0[idx]
            self._draw_tile(board, row, col, self.COLOR_PLAYER_0, self.COLOR_BLACK)
    
    def _draw_indices(self, board: np.ndarray) -> None:
        """Draw index numbers on all track positions (skip 0 - off board)."""
        for idx in range(1, len(TRACK_PLAYER_0)):
            row, col = TRACK_PLAYER_0[idx]
            # Skip off-board positions
            if row == -1 or col == -1:
                continue
            # Show visual index (analyzer index - 1) since 0 is off-board
            visual_idx = idx - 1
            self._draw_text_at_tile(board, str(visual_idx), row, col, 
                                   self.COLOR_WHITE, font_scale=0.6, thickness=2,
                                   bg_color=self.COLOR_BLACK)
    
    def _add_title(self, img: np.ndarray) -> np.ndarray:
        """Add title to the board image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Ludo Board - Index Reference (Player 0 Perspective)"
        font_scale = 1.2
        font_thickness = 3
        
        # Get text size
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        
        # Calculate position (centered at top)
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 40
        
        # Draw background
        cv2.rectangle(
            img,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            self.COLOR_BLACK,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            title,
            (text_x, text_y),
            font,
            font_scale,
            self.COLOR_WHITE,
            font_thickness
        )
        
        return img
    
    def _add_legend(self, img: np.ndarray) -> np.ndarray:
        """Add legend showing position types."""
        # Convert analyzer indices to visual indices (subtract 1 since 0 is off-board)
        visual_stars = [idx - 1 for idx in self.STARS]
        visual_globes = [idx - 1 for idx in self.GLOBES]
        
        legend_entries = [
            ("HOME YARD", self.COLOR_PLAYER_0, "Off board (needs 6 to start)"),
            ("TRACK (0-49)", self.COLOR_TILE, "Main track around board"),
            ("STAR", self.COLOR_TILE, f"Indices: {', '.join(map(str, visual_stars))}"),
            ("GLOBE (Safe)", self.COLOR_TILE, f"Indices: {', '.join(map(str, visual_globes))}"),
            ("HOME STRETCH (50-54)", self.COLOR_PLAYER_0, "Safe zone to goal"),
            ("GOAL (55)", self.COLOR_GOAL, "Win condition"),
        ]
        
        # Legend dimensions
        legend_width = 480
        legend_height = len(legend_entries) * 60 + 40
        legend_padding = 20
        
        # Legend position (bottom-right)
        legend_x = img.shape[1] - legend_width - legend_padding
        legend_y = img.shape[0] - legend_height - legend_padding
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            self.COLOR_BLACK,
            -1
        )
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Draw border
        cv2.rectangle(
            img,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            self.COLOR_WHITE,
            2
        )
        
        # Draw title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            "LEGEND",
            (legend_x + 10, legend_y + 25),
            font,
            0.7,
            self.COLOR_WHITE,
            2
        )
        
        # Draw entries
        entry_y_start = legend_y + 50
        for i, (label, color, description) in enumerate(legend_entries):
            entry_y = entry_y_start + (i * 60)
            
            # Color box
            box_size = 15
            box_x = legend_x + 15
            box_y = entry_y - box_size
            cv2.rectangle(img, (box_x, box_y), (box_x + box_size, box_y + box_size), color, -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_size, box_y + box_size), self.COLOR_WHITE, 1)
            
            # Label
            label_x = box_x + box_size + 10
            cv2.putText(img, label, (label_x, entry_y), font, 0.5, self.COLOR_WHITE, 1)
            
            # Description
            cv2.putText(img, description, (label_x, entry_y + 16), font, 0.35, (180, 180, 180), 1)
        
        return img
    
    def show(self, wait_key: int = 0) -> None:
        """Display the board in a window."""
        board_img = self.render_annotated_board()
        cv2.imshow(self.window_name, board_img)
        cv2.waitKey(wait_key)
    
    def save(self, filepath: str) -> None:
        """Save the board to a file."""
        board_img = self.render_annotated_board()
        cv2.imwrite(filepath, board_img)
        print(f"Board visualization saved to: {filepath}")
    
    def close(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()


def visualize_board_indices(save_path: Optional[str] = None, show: bool = True,
                           width: Optional[int] = None, height: Optional[int] = None,
                           show_legend: bool = True,
                           show_indices: bool = True) -> None:
    
    """
    Convenience function to visualize board indices.
    
    Args:
        save_path: Optional path to save the visualization
        show: Whether to display in a window
        width: Target width in pixels (None = use default)
        height: Target height in pixels (None = use default)
    """
    if not HAS_CV2:
        print("Error: opencv-python (cv2) is required for visualization")
        return
    
    try:
        # Get screen resolution if not specified
        if width is None and height is None:
            try:
                # Try to get screen resolution
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                
                # Use 80% of screen size
                width = int(screen_width * 0.8)
                height = int(screen_height * 0.8)
                
                # Maintain aspect ratio
                aspect_ratio = BOARD_SIZE[0] / BOARD_SIZE[1]  # height / width
                if height / width > aspect_ratio:
                    height = int(width * aspect_ratio)
                else:
                    width = int(height / aspect_ratio)
            except:
                # Fallback to default size
                width = 1280
                height = 1088
        
        visualizer = BoardVisualizer(target_width=width, target_height=height, show_indices=show_indices, show_legend=show_legend)
        
        if save_path:
            visualizer.save(save_path)
        
        if show:
            print(f"Displaying board visualization ({width}x{height})...")
            print("Press any key to close the window.")
            visualizer.show(wait_key=0)
        
        visualizer.close()
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


