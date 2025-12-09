"""
Standard Ludo Board Visualizer for Curriculum Levels.
Creates a proper cross-shaped Ludo board matching the classic game layout.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
import os
from collections import defaultdict

class StandardLudoBoardVisualizer:
    """Visualizer using standard Ludo board layout."""

    def __init__(self, level: int, num_players: int = 2, tokens_per_player: int = 1):
        """
        Initialize visualizer with standard Ludo board.

        Args:
            level: Curriculum level (1-5)
            num_players: Number of players (2 or 4)
            tokens_per_player: Tokens per player (1 or 2)
        """
        self.level = level
        self.num_players = num_players
        self.tokens_per_player = tokens_per_player

        # Window settings
        self.window_name = f"Level {level} - Ludo Curriculum (Standard Board)"
        
        # Board dimensions (17 rows x 20 cols of 64x64 tiles)
        self.tile_size = 64
        self.board_rows = 17
        self.board_cols = 20
        self.board_width = self.board_cols * self.tile_size
        self.board_height = self.board_rows * self.tile_size
        
        # Piece settings
        self.piece_radius = 20

        # Colors (BGR format for OpenCV)
        self.BG_COLOR = (186, 202, 215)  # Light blue-gray
        self.TRACK_COLOR = (174, 95, 74)  # Brown
        self.GOAL_COLOR = (234, 255, 0)  # Gold
        self.TEXT_COLOR = (0, 0, 0)  # Black
        
        # Player colors (matching standard Ludo)
        self.PLAYER_COLORS = {
            0: (39, 214, 74),   # Player 0 (Agent) - Green
            1: (105, 219, 210), # Player 1 - Yellow
            2: (46, 171, 255),  # Player 2 - Blue
            3: (255, 107, 107), # Player 3 - Red
        }
        
        # Player area colors (lighter versions)
        self.PLAYER_AREA_COLORS = {
            0: (173, 185, 58),  # Green area
            1: (241, 184, 53),  # Yellow area
            2: (79, 78, 127),   # Blue area
            3: (239, 59, 74),   # Red area
        }

        # Define board layout using grid coordinates [row, col]
        self._setup_board_layout()
        
        # Load assets
        self._load_assets()
        
        # Mapping from linear position (0-59) to board tile coordinates
        self._create_position_mappings()

    def _setup_board_layout(self):
        """Define the standard Ludo board layout."""
        
        # Goal area (center 3x3)
        self.goal_tiles = [
            [7, 7], [7, 8], [7, 9],
            [8, 7], [8, 8], [8, 9],
            [9, 7], [9, 8], [9, 9]
        ]
        
        # Player home tiles (starting areas)
        self.home_tiles = {
            0: [[12, 3], [14, 3], [12, 5], [14, 5]],  # Player 0 (bottom-left)
            1: [[3, 2], [5, 2], [3, 4], [5, 4]],       # Player 1 (top-left)
            2: [[2, 11], [2, 13], [4, 11], [4, 13]],   # Player 2 (top-right)
            3: [[11, 12], [13, 12], [11, 14], [13, 14]] # Player 3 (bottom-right)
        }
        
        # Player goal paths (leading to center)
        self.goal_paths = {
            0: [[14, 8], [13, 8], [12, 8], [11, 8], [10, 8], [9, 8]],
            1: [[8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7]],
            2: [[2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]],
            3: [[8, 14], [8, 13], [8, 12], [8, 11], [8, 10], [8, 9]]
        }
        
        # Globe positions (safe zones)
        self.globe_tiles = [
            [14, 7], [7, 2], [2, 9], [9, 14],  # Start positions
            [13, 9], [9, 3], [3, 7], [7, 13]   # Middle globes
        ]
        
        # Star positions (bonus tiles)
        self.star_tiles = [
            [15, 8], [8, 1], [1, 8], [8, 15],
            [10, 7], [7, 6], [6, 9], [9, 10]
        ]
        
        # Main track tiles (the cross-shaped path)
        self.track_tiles = [
            # Horizontal tracks
            [10, 9], [11, 9], [12, 9], [13, 9], [14, 9], [15, 9],
            [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7],
            [1, 9], [3, 9], [4, 9], [5, 9],
            [10, 7], [11, 7], [12, 7], [13, 7], [15, 7],
            # Vertical tracks
            [7, 1], [7, 3], [7, 4], [7, 5], [7, 6],
            [9, 1], [9, 2], [9, 4], [9, 5], [9, 6],
            [7, 10], [7, 11], [7, 12], [7, 14], [7, 15],
            [9, 11], [9, 12], [9, 13], [9, 15]
        ]

        # Standard track for each player (57 positions + goal)
        self._define_player_tracks()

    def _define_player_tracks(self):
        """Define the complete track path for each player."""
        # Player 0 track (starts at [14,7])
        track_p0 = np.array([
            [14, 7], [13, 7], [12, 7], [11, 7], [10, 7], [9, 6], [9, 5], [9, 4], [9, 3], 
            [9, 2], [9, 1], [8, 1], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], 
            [6, 7], [5, 7], [4, 7], [3, 7], [2, 7], [1, 7], [1, 8], [1, 9], [2, 9],
            [3, 9], [4, 9], [5, 9], [6, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], 
            [7, 15], [8, 15], [9, 15], [9, 14], [9, 13], [9, 12], [9, 11], [9, 10], 
            [10, 9], [11, 9], [12, 9], [13, 9], [14, 9], [15, 9], [15, 8], [14, 8],
            [13, 8], [12, 8], [11, 8], [10, 8], [9, 8]
        ])
        
        # Calculate other player tracks by rotation
        diff_1 = track_p0 - track_p0[0]
        diff_2 = np.array([diff_1[:, 1], -diff_1[:, 0]]).T
        diff_3 = np.array([-diff_1[:, 0], -diff_1[:, 1]]).T
        diff_4 = np.array([-diff_1[:, 1], diff_1[:, 0]]).T
        
        self.player_tracks = {
            0: track_p0,
            1: np.array([7, 2]) + diff_2,
            2: np.array([2, 9]) + diff_3,
            3: np.array([9, 14]) + diff_4
        }

    def _load_assets(self):
        """Load star and globe images."""
        try:
            folder = os.path.dirname(__file__)
            assets_folder = os.path.join(folder, '../ludo/assets')
            
            # Load globe
            glob_img = cv2.imread(os.path.join(assets_folder, 'glob.png'))
            if glob_img is not None:
                self.globe_img = cv2.resize(glob_img, (40, 40))
                self.globe_mask = cv2.inRange(self.globe_img, (255, 255, 255), (255, 255, 255)) == 0
            else:
                self.globe_img = None
                
            # Load star
            star_img = cv2.imread(os.path.join(assets_folder, 'star.png'))
            if star_img is not None:
                self.star_img = cv2.resize(star_img, (40, 40))
                self.star_mask = cv2.inRange(self.star_img, (255, 255, 255), (255, 255, 255)) == 0
            else:
                self.star_img = None
                
        except Exception as e:
            print(f"Warning: Could not load assets: {e}")
            self.globe_img = None
            self.star_img = None

    def _create_position_mappings(self):
        """Map linear positions (0-59) to board coordinates."""
        self.position_to_tile = {}
        
        for player in range(4):
            self.position_to_tile[player] = {}
            
            # Position 0 = home
            self.position_to_tile[player][0] = self.home_tiles[player]
            
            # Positions 1-57 = track
            for i, tile in enumerate(self.player_tracks[player], start=1):
                self.position_to_tile[player][i] = tuple(tile)
            
            # Position 58-60 = goal
            for i in range(58, 61):
                self.position_to_tile[player][i] = (8, 8)  # Center goal

    def render(self, state_info: dict):
        """
        Render the current game state.

        Args:
            state_info: Dictionary containing:
                - player_positions: List of token positions for each player
                - current_player: Current player index
                - dice: Current dice value (if applicable)
                - winner: Winner index (if game over)
                - step: Current step number
        """
        # Create board
        board = self._draw_board()
        
        # Draw header with game info
        self._draw_header_panel(board, state_info)
        
        # Draw pieces
        self._draw_all_pieces(board, state_info.get('player_positions', []))
        
        # Draw legend
        self._draw_legend(board)
        
        # Draw status
        self._draw_status(board, state_info)
        
        # Show window
        cv2.imshow(self.window_name, board)
        cv2.waitKey(1)

    def _draw_board(self):
        """Draw the base Ludo board."""
        board = np.full((self.board_height, self.board_width, 3), 
                       self.BG_COLOR, dtype=np.uint8)
        
        # Draw goal area (center)
        self._draw_multi_tiles(board, (7, 7), (9, 9), 
                              line_color=self.TEXT_COLOR, 
                              fill_color=self.GOAL_COLOR)
        
        # Draw track tiles
        for tile in self.track_tiles + self.star_tiles + self.globe_tiles:
            self._draw_tile(board, tile[0], tile[1], 
                          line_color=self.TEXT_COLOR, 
                          fill_color=self.TRACK_COLOR)
        
        # Draw player home areas and goal paths
        for player in range(self.num_players):
            area_color = self.PLAYER_AREA_COLORS[player]
            
            # Home tiles
            for tile in self.home_tiles[player]:
                self._draw_tile(board, tile[0], tile[1], 
                              line_color=self.TEXT_COLOR, 
                              fill_color=area_color)
            
            # Goal path
            for tile in self.goal_paths[player]:
                self._draw_tile(board, tile[0], tile[1], 
                              line_color=self.TEXT_COLOR, 
                              fill_color=area_color)
        
        # Draw globes
        if self.globe_img is not None:
            for tile in self.globe_tiles:
                self._put_image_at_tile(board, self.globe_img, 
                                       tile[0], tile[1], self.globe_mask)
        
        # Draw stars
        if self.star_img is not None:
            for tile in self.star_tiles:
                self._put_image_at_tile(board, self.star_img, 
                                       tile[0], tile[1], self.star_mask)
        
        return board

    def _draw_tile(self, board, row, col, line_color=None, fill_color=None, thickness=2):
        """Draw a single tile."""
        top_left = (col * self.tile_size, row * self.tile_size)
        bot_right = ((col + 1) * self.tile_size, (row + 1) * self.tile_size)
        
        if fill_color is not None:
            cv2.rectangle(board, top_left, bot_right, fill_color, -1)
        if line_color is not None:
            cv2.rectangle(board, top_left, bot_right, line_color, thickness)

    def _draw_multi_tiles(self, board, top_left_tile, bottom_right_tile, 
                          line_color=None, fill_color=None, thickness=2):
        """Draw a rectangular region of tiles."""
        top_left = (top_left_tile[1] * self.tile_size, top_left_tile[0] * self.tile_size)
        bot_right = ((bottom_right_tile[1] + 1) * self.tile_size, 
                    (bottom_right_tile[0] + 1) * self.tile_size)
        
        if fill_color is not None:
            cv2.rectangle(board, top_left, bot_right, fill_color, -1)
        if line_color is not None:
            cv2.rectangle(board, top_left, bot_right, line_color, thickness)

    def _get_tile_center(self, row, col):
        """Get the center pixel coordinates of a tile."""
        x = col * self.tile_size + self.tile_size // 2
        y = row * self.tile_size + self.tile_size // 2
        return (x, y)

    def _put_image_at_tile(self, board, image, row, col, mask=None):
        """Place an image at a tile center."""
        if mask is None:
            mask = np.full(image.shape[:2], True)
        
        center = self._get_tile_center(row, col)
        h, w = image.shape[:2]
        
        y1 = max(0, center[1] - h // 2)
        y2 = min(board.shape[0], center[1] + h // 2)
        x1 = max(0, center[0] - w // 2)
        x2 = min(board.shape[1], center[0] + w // 2)
        
        if y2 > y1 and x2 > x1:
            board[y1:y2, x1:x2][mask] = image[mask]

    def _draw_header_panel(self, board, state_info):
        """Draw header panel with game info."""
        # Panel background
        panel_height = 80
        cv2.rectangle(board, (0, 0), (self.board_width, panel_height), 
                     (50, 50, 50), -1)
        
        # Title
        title = f"Level {self.level} - Curriculum Learning (Standard Board)"
        cv2.putText(board, title, (20, 35), cv2.FONT_HERSHEY_DUPLEX,
                   0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Game info
        step = state_info.get('step', 0)
        current = state_info.get('current_player', 0)
        dice = state_info.get('dice', None)
        
        info_y = 65
        cv2.putText(board, f"Step: {step}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(board, f"Player: {current}", (180, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.PLAYER_COLORS[current], 2, cv2.LINE_AA)
        
        if dice is not None:
            cv2.putText(board, f"Dice: {dice}", (350, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_all_pieces(self, board, player_positions):
        """Draw all player pieces on the board."""
        # Group pieces by tile to handle overlapping
        tiles_in_use = defaultdict(lambda: [])
        
        for player_idx, tokens in enumerate(player_positions):
            if player_idx >= self.num_players:
                continue
                
            for token_idx, pos in enumerate(tokens):
                tile = self._get_tile_for_position(player_idx, pos)
                if tile:
                    tiles_in_use[tile].append((player_idx, token_idx))
        
        # Draw pieces
        for tile, pieces in tiles_in_use.items():
            self._draw_pieces_at_tile(board, tile[0], tile[1], pieces)

    def _get_tile_for_position(self, player, position):
        """Convert a position to board tile coordinates."""
        if position == 0:
            # Token at home - use first home tile
            return tuple(self.home_tiles[player][0])
        elif position >= 60:
            # Token at goal
            return (8, 8)  # Center
        elif 1 <= position <= 57:
            # Token on track
            track_idx = position - 1
            if track_idx < len(self.player_tracks[player]):
                tile = self.player_tracks[player][track_idx]
                return tuple(tile)
        return None

    def _draw_pieces_at_tile(self, board, row, col, pieces):
        """Draw multiple pieces at a single tile."""
        center = self._get_tile_center(row, col)
        
        if len(pieces) == 1:
            # Single piece - draw at center
            player_idx, token_idx = pieces[0]
            self._draw_piece(board, center, player_idx, token_idx)
        else:
            # Multiple pieces - offset them
            offsets = [(-12, -12), (12, -12), (-12, 12), (12, 12)]
            for i, (player_idx, token_idx) in enumerate(pieces[:4]):
                offset = offsets[i] if i < len(offsets) else (0, 0)
                pos = (center[0] + offset[0], center[1] + offset[1])
                self._draw_piece(board, pos, player_idx, token_idx, radius=15)

    def _draw_piece(self, board, center, player_idx, token_idx, radius=None):
        """Draw a single piece."""
        if radius is None:
            radius = self.piece_radius
            
        color = self.PLAYER_COLORS[player_idx]
        
        # Shadow
        cv2.circle(board, (center[0] + 2, center[1] + 2), radius, (0, 0, 0), -1)
        
        # Piece
        cv2.circle(board, center, radius, color, -1)
        cv2.circle(board, center, radius, (0, 0, 0), 2)
        
        # Token number
        cv2.putText(board, str(token_idx), (center[0] - 6, center[1] + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_legend(self, board):
        """Draw legend with player colors and game elements."""
        # Legend position (bottom-right corner)
        legend_x = self.board_width - 350
        legend_y = self.board_height - 300
        legend_width = 330
        legend_height = 280
        
        # Background
        cv2.rectangle(board, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), -1)
        cv2.rectangle(board, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (0, 0, 0), 3)
        
        # Title - matching heading font style
        title_y = legend_y + 35
        cv2.putText(board, "Legend:", (legend_x + 15, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Players section
        y_offset = 60
        for i in range(self.num_players):
            y_pos = legend_y + y_offset + i * 40
            color = self.PLAYER_COLORS[i]
            
            # Color circle
            cv2.circle(board, (legend_x + 25, y_pos), 15, color, -1)
            cv2.circle(board, (legend_x + 25, y_pos), 15, (0, 0, 0), 2)
            
            # Label
            label = f"Player {i}" + (" (Agent)" if i == 0 else " (AI)")
            cv2.putText(board, label, (legend_x + 50, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Special tiles section
        y_pos = legend_y + y_offset + self.num_players * 40 + 20
        
        # Globe (safe zone)
        if self.globe_img is not None:
            self._put_image_at_tile_offset(board, self.globe_img, 
                                          legend_x + 25, y_pos, self.globe_mask)
        else:
            cv2.circle(board, (legend_x + 25, y_pos), 15, (50, 200, 50), -1)
            cv2.circle(board, (legend_x + 25, y_pos), 15, (0, 0, 0), 2)
        
        cv2.putText(board, "Safe Zone (Globe)", (legend_x + 50, y_pos + 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        y_pos += 40
        
        # Star (bonus)
        if self.star_img is not None:
            self._put_image_at_tile_offset(board, self.star_img, 
                                          legend_x + 25, y_pos, self.star_mask)
        else:
            # Draw simple star shape
            pts = np.array([[0, -12], [3, -3], [12, 0], [3, 3], [0, 12], 
                           [-3, 3], [-12, 0], [-3, -3]], np.int32)
            pts = pts + [legend_x + 25, y_pos]
            cv2.fillPoly(board, [pts], (0, 215, 255))
        
        cv2.putText(board, "Star (Safe)", (legend_x + 50, y_pos + 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    def _put_image_at_tile_offset(self, board, image, x, y, mask=None):
        """Place an image at specific pixel coordinates."""
        if mask is None:
            mask = np.full(image.shape[:2], True)
        
        h, w = image.shape[:2]
        
        y1 = max(0, y - h // 2)
        y2 = min(board.shape[0], y + h // 2)
        x1 = max(0, x - w // 2)
        x2 = min(board.shape[1], x + w // 2)
        
        if y2 > y1 and x2 > x1:
            img_y1 = h // 2 - (y - y1)
            img_y2 = h // 2 + (y2 - y)
            img_x1 = w // 2 - (x - x1)
            img_x2 = w // 2 + (x2 - x)
            
            if img_y2 > img_y1 and img_x2 > img_x1:
                board[y1:y2, x1:x2][mask[img_y1:img_y2, img_x1:img_x2]] = \
                    image[img_y1:img_y2, img_x1:img_x2][mask[img_y1:img_y2, img_x1:img_x2]]

    def _draw_status(self, board, state_info):
        """Draw game status."""
        winner = state_info.get('winner', None)
        
        if winner is not None:
            msg = f"GAME OVER - Player {winner} WINS!"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
            
            # Center the banner
            box_x = (self.board_width - text_size[0]) // 2 - 30
            box_y = self.board_height - 120
            
            # Shadow
            cv2.rectangle(board, (box_x + 5, box_y + 5),
                         (box_x + text_size[0] + 65, box_y + 85),
                         (0, 0, 0), -1)
            
            # Banner
            cv2.rectangle(board, (box_x, box_y),
                         (box_x + text_size[0] + 60, box_y + 80),
                         self.GOAL_COLOR, -1)
            cv2.rectangle(board, (box_x, box_y),
                         (box_x + text_size[0] + 60, box_y + 80),
                         (0, 0, 0), 4)
            
            # Text
            cv2.putText(board, msg, (box_x + 30, box_y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)

    def close(self):
        """Close visualization window."""
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except:
            pass

