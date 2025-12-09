"""
Enhanced CV2 Visualizer for Curriculum Levels - Ludo Board Style.
Creates beautiful Ludo-style boards for all 5 curriculum levels.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import os


class EnhancedCurriculumVisualizer:
    """Enhanced visualizer with proper Ludo board styling."""

    def __init__(self, level: int, num_players: int = 2, tokens_per_player: int = 1):
        """
        Initialize enhanced visualizer.

        Args:
            level: Curriculum level (1-5)
            num_players: Number of players
            tokens_per_player: Tokens per player
        """
        self.level = level
        self.num_players = num_players
        self.tokens_per_player = tokens_per_player

        # Window settings
        self.window_name = f"Level {level} - Ludo Curriculum"

        # Tile settings (like original Ludo visualizer)
        self.tile_size = 50  # Pixels per tile
        self.piece_radius = 18

        # Board dimensions based on level
        if level <= 2:
            # Simple track: 0-60 positions + info area
            self.board_width = 70 * self.tile_size  # 70 tiles wide
            self.board_height = 12 * self.tile_size  # 12 tiles high
        else:
            # Multi-token/multi-player: grid layout
            self.board_width = 70 * self.tile_size
            self.board_height = (4 + num_players * 3) * self.tile_size

        # Colors (BGR format, similar to original Ludo)
        self.BG_COLOR = (186, 202, 215)  # Light blue-gray
        self.TRACK_COLOR = (174, 95, 74)  # Brown
        self.GOAL_COLOR = (0, 255, 234)  # Gold (BGR)
        self.TEXT_COLOR = (0, 0, 0)  # Black
        self.SAFE_ZONE_COLOR = (100, 180, 255)  # Light orange

        # Player colors (vibrant, like original Ludo)
        self.PLAYER_COLORS = [
            (39, 214, 74),   # Player 0 (Agent) - Green
            (105, 219, 210), # Player 1 - Yellow
            (255, 171, 46),  # Player 2 - Blue
            (107, 107, 255), # Player 3 - Red
        ]

        # Player area colors (lighter versions)
        self.PLAYER_AREA_COLORS = [
            (58, 185, 173),  # Green area
            (53, 184, 241),  # Yellow area
            (127, 78, 79),   # Blue area
            (74, 59, 239),   # Red area
        ]

        # Track settings
        self.track_length = 60

        # Safe zones - cannot be captured here (actual game safe zones)
        self.safe_zones = [10, 20, 30, 40, 50] if level >= 3 else []
        
        # Safe zones (Globes) - for visual styling (levels 1-2)
        self.globe_positions = [0, 8, 21, 34, 47]  # Start + strategic positions

        # Star positions - bonus movement positions
        self.star_positions = [14, 27, 40, 53]  # Between globes

        # Load assets if available
        self.assets_loaded = False
        try:
            assets_path = os.path.join(os.path.dirname(__file__), '../ludo/assets')
            star_img = cv2.imread(os.path.join(assets_path, 'star.png'))
            if star_img is not None:
                self.star_img = cv2.resize(star_img, (30, 30))
                self.star_mask = cv2.inRange(self.star_img, (255, 255, 255), (255, 255, 255)) == 0
                self.assets_loaded = True
        except:
            pass

    def render(self, state_info: dict):
        """Render the current game state with enhanced graphics."""
        # Create board
        img = self._create_board()

        # Draw title and info
        self._draw_header(img, state_info)

        # Draw game board based on level
        if self.level <= 2:
            self._draw_horizontal_track(img, state_info)
        else:
            self._draw_multi_track_board(img, state_info)

        # Draw game status
        self._draw_status(img, state_info)

        # Show window
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def _create_board(self):
        """Create the base board."""
        img = np.full((self.board_height, self.board_width, 3),
                     self.BG_COLOR, dtype=np.uint8)
        return img

    def _draw_header(self, img, state_info):
        """Draw header with game info."""
        header_height = 2 * self.tile_size

        # Title background
        cv2.rectangle(img, (0, 0), (self.board_width, header_height),
                     (50, 50, 50), -1)

        # Title
        title = f"Level {self.level} - Curriculum Learning"
        cv2.putText(img, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX,
                   1.3, (255, 255, 255), 2, cv2.LINE_AA)

        # Game info
        step = state_info.get('step', 0)
        current = state_info.get('current_player', 0)
        dice = state_info.get('dice', None)

        info_y = 85
        cv2.putText(img, f"Step: {step}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, f"Current: Player {current}", (200, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.PLAYER_COLORS[current], 2, cv2.LINE_AA)

        if dice is not None:
            cv2.putText(img, f"Dice:", (450, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            self._draw_dice(img, dice, (540, info_y - 25))

    def _draw_dice(self, img, dice: int, pos: Tuple[int, int]):
        """Draw dice with dots."""
        x, y = pos
        size = 50

        # Dice box with shadow
        cv2.rectangle(img, (x + 3, y + 3), (x + size + 3, y + size + 3),
                     (0, 0, 0), -1)  # Shadow
        cv2.rectangle(img, (x, y), (x + size, y + size),
                     (255, 255, 255), -1)  # White dice
        cv2.rectangle(img, (x, y), (x + size, y + size),
                     (0, 0, 0), 2)  # Border

        # Dot positions
        dot_positions = {
            1: [(25, 25)],
            2: [(15, 15), (35, 35)],
            3: [(15, 15), (25, 25), (35, 35)],
            4: [(15, 15), (35, 15), (15, 35), (35, 35)],
            5: [(15, 15), (35, 15), (25, 25), (15, 35), (35, 35)],
            6: [(15, 15), (35, 15), (15, 25), (35, 25), (15, 35), (35, 35)],
        }

        for dx, dy in dot_positions.get(dice, []):
            cv2.circle(img, (x + dx, y + dy), 5, (0, 0, 0), -1)

    def _draw_horizontal_track(self, img, state_info):
        """Draw horizontal track for Levels 1-2."""
        positions = state_info.get('player_positions', [[0], [0]])

        # Track area
        track_y = 4 * self.tile_size
        track_start_x = 2 * self.tile_size

        # Draw each tile on the track
        for i in range(self.track_length + 1):
            x = track_start_x + i * self.tile_size

            # Determine tile color
            if i == self.track_length:
                tile_color = self.GOAL_COLOR
            elif i in self.globe_positions:
                tile_color = (200, 255, 200)  # Light green for globes
            elif i in self.star_positions:
                tile_color = (255, 200, 150)  # Light orange for stars
            else:
                tile_color = self.TRACK_COLOR

            # Draw tile
            cv2.rectangle(img, (x, track_y),
                         (x + self.tile_size, track_y + 3 * self.tile_size),
                         tile_color, -1)
            cv2.rectangle(img, (x, track_y),
                         (x + self.tile_size, track_y + 3 * self.tile_size),
                         (0, 0, 0), 2)

            # Draw globe marker
            if i in self.globe_positions:
                self._draw_globe(img, x + self.tile_size // 2,
                                track_y + 3 * self.tile_size // 2)

            # Draw star marker
            elif i in self.star_positions:
                self._draw_star(img, x + self.tile_size // 2,
                               track_y + 3 * self.tile_size // 2)

            # Position label
            if i % 10 == 0:
                label = str(i) if i < self.track_length else "GOAL"
                cv2.putText(img, label, (x + 5, track_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw home area
        home_x = track_start_x - self.tile_size
        cv2.rectangle(img, (home_x, track_y),
                     (home_x + self.tile_size, track_y + 3 * self.tile_size),
                     (150, 150, 150), -1)
        cv2.rectangle(img, (home_x, track_y),
                     (home_x + self.tile_size, track_y + 3 * self.tile_size),
                     (0, 0, 0), 2)
        cv2.putText(img, "HOME", (home_x + 2, track_y + self.tile_size),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw pieces
        self._draw_pieces_on_track(img, positions, track_start_x, track_y)

        # Draw legend
        self._draw_legend_simple(img, track_start_x, track_y + 5 * self.tile_size)

    def _draw_pieces_on_track(self, img, positions, track_start_x, track_y):
        """Draw player pieces on the track."""
        for player_idx, player_tokens in enumerate(positions):
            color = self.PLAYER_COLORS[player_idx]

            for token_idx, pos in enumerate(player_tokens):
                # Calculate position
                if pos < 0:  # Home
                    x = track_start_x - self.tile_size + self.tile_size // 2
                elif pos >= self.track_length:  # Goal
                    x = track_start_x + self.track_length * self.tile_size + self.tile_size // 2
                else:
                    x = track_start_x + int(pos * self.tile_size) + self.tile_size // 2

                # Y position (offset for multiple tokens)
                y_offset = (token_idx - len(player_tokens) / 2 + 0.5) * 30
                y = int(track_y + 3 * self.tile_size // 2 + y_offset)

                # Draw piece with shadow
                cv2.circle(img, (x + 2, y + 2), self.piece_radius,
                          (0, 0, 0), -1)  # Shadow
                cv2.circle(img, (x, y), self.piece_radius, color, -1)
                cv2.circle(img, (x, y), self.piece_radius, (0, 0, 0), 2)

                # Player label
                cv2.putText(img, f"P{player_idx}", (x - 12, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_multi_track_board(self, img, state_info):
        """Draw multi-track board for Levels 3-5."""
        positions = state_info.get('player_positions', [])

        start_y = 3 * self.tile_size
        track_spacing = 3 * self.tile_size
        track_start_x = 3 * self.tile_size

        for player_idx, player_tokens in enumerate(positions):
            y_pos = start_y + player_idx * track_spacing
            color = self.PLAYER_COLORS[player_idx]
            area_color = self.PLAYER_AREA_COLORS[player_idx]

            # Player label area
            label_x = self.tile_size
            cv2.rectangle(img, (label_x, y_pos),
                         (track_start_x - 10, y_pos + 2 * self.tile_size),
                         area_color, -1)
            cv2.rectangle(img, (label_x, y_pos),
                         (track_start_x - 10, y_pos + 2 * self.tile_size),
                         (0, 0, 0), 2)

            label = f"P{player_idx}"
            if player_idx == 0:
                label += "\n(Agent)"
            cv2.putText(img, f"Player {player_idx}",
                       (label_x + 5, y_pos + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if player_idx == 0:
                cv2.putText(img, "(Agent)",
                           (label_x + 10, y_pos + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Draw track tiles
            for i in range(self.track_length + 1):
                x = track_start_x + i * self.tile_size

                # Tile color - use actual safe zones for levels 3-5
                if i == self.track_length:
                    tile_color = self.GOAL_COLOR
                elif i in self.safe_zones:
                    tile_color = (100, 255, 100)  # Light green for safe zones
                elif i in self.globe_positions and self.level <= 2:
                    tile_color = (200, 255, 200)  # Light green for globes (levels 1-2)
                elif i in self.star_positions and self.level <= 2:
                    tile_color = (255, 200, 150)  # Light orange for stars (levels 1-2)
                else:
                    tile_color = self.TRACK_COLOR

                # Draw tile
                cv2.rectangle(img, (x, y_pos),
                             (x + self.tile_size, y_pos + 2 * self.tile_size),
                             tile_color, -1)
                cv2.rectangle(img, (x, y_pos),
                             (x + self.tile_size, y_pos + 2 * self.tile_size),
                             (0, 0, 0), 1)

                # Draw markers - use safe zones for levels 3-5
                if i in self.safe_zones:
                    self._draw_globe(img, x + self.tile_size // 2,
                                    y_pos + self.tile_size)
                elif i in self.globe_positions and self.level <= 2:
                    self._draw_globe(img, x + self.tile_size // 2,
                                    y_pos + self.tile_size)
                elif i in self.star_positions and self.level <= 2:
                    self._draw_star(img, x + self.tile_size // 2,
                                   y_pos + self.tile_size)
                
                # Draw subtle capture zone markers (red dots) for vulnerable positions
                if self.level >= 3 and i not in self.safe_zones and i > 0 and i < self.track_length:
                    if i % 5 == 0:  # Every 5th position to avoid clutter
                        cv2.circle(img, (x + self.tile_size // 2, y_pos + self.tile_size), 
                                 3, (0, 0, 255), -1)  # Red dot

            # Draw home
            home_x = track_start_x - self.tile_size
            cv2.rectangle(img, (home_x, y_pos),
                         (home_x + self.tile_size, y_pos + 2 * self.tile_size),
                         area_color, -1)
            cv2.rectangle(img, (home_x, y_pos),
                         (home_x + self.tile_size, y_pos + 2 * self.tile_size),
                         (0, 0, 0), 2)

            # Draw tokens
            for token_idx, pos in enumerate(player_tokens):
                if pos < 0:  # Home
                    x = home_x + self.tile_size // 2
                elif pos >= self.track_length:  # Goal
                    x = track_start_x + self.track_length * self.tile_size + self.tile_size // 2
                else:
                    x = track_start_x + int(pos * self.tile_size) + self.tile_size // 2

                y = y_pos + self.tile_size + (token_idx - 0.5) * 15

                # Draw piece
                cv2.circle(img, (x + 2, int(y + 2)), self.piece_radius - 3,
                          (0, 0, 0), -1)  # Shadow
                cv2.circle(img, (x, int(y)), self.piece_radius - 3, color, -1)
                cv2.circle(img, (x, int(y)), self.piece_radius - 3, (0, 0, 0), 2)

                # Token number
                cv2.putText(img, str(token_idx), (x - 6, int(y + 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw legend for multi-token boards (positioned on the right side)
        legend_x = track_start_x + (self.track_length + 2) * self.tile_size
        legend_y = start_y
        self._draw_legend_multi(img, legend_x, legend_y)

    def _draw_legend_multi(self, img, x: int, y: int):
        """Draw legend for multi-token boards with improved font matching heading style."""
        legend_width = 12 * self.tile_size
        legend_height = (self.num_players + 3) * self.tile_size if self.safe_zones else (self.num_players + 1) * self.tile_size

        # Background box
        cv2.rectangle(img, (x, y), (x + legend_width, y + legend_height), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), 3)

        # Title - use same font as heading (FONT_HERSHEY_DUPLEX)
        title_y = y + 35
        cv2.putText(img, "Legend:", (x + 15, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # Players section
        y_offset = 60
        for i in range(self.num_players):
            y_pos = y + y_offset + i * 40
            color = self.PLAYER_COLORS[i]

            # Draw color circle
            cv2.circle(img, (x + 25, y_pos), 15, color, -1)
            cv2.circle(img, (x + 25, y_pos), 15, (0, 0, 0), 2)

            # Draw label with matching font style
            label = f"Player {i}" + (" (Agent)" if i == 0 else " (Random)")
            cv2.putText(img, label, (x + 50, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        # Safe zone and capture zone legend if applicable
        if self.safe_zones:
            y_pos = y + y_offset + self.num_players * 40 + 20
            
            # Safe zone legend
            self._draw_globe(img, x + 25, y_pos)
            cv2.putText(img, "Safe Zone (No Capture)", (x + 50, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
            y_pos += 40
            
            # Capture zone legend
            cv2.circle(img, (x + 25, y_pos), 5, (0, 0, 255), -1)  # Red dot
            cv2.circle(img, (x + 25, y_pos), 5, (0, 0, 0), 1)
            cv2.putText(img, "Capture Zone (Vulnerable)", (x + 50, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_star(self, img, x: int, y: int):
        """Draw a star marker."""
        if self.assets_loaded:
            # Use actual star image
            try:
                y1 = max(0, y - 15)
                y2 = min(img.shape[0], y + 15)
                x1 = max(0, x - 15)
                x2 = min(img.shape[1], x + 15)

                h = y2 - y1
                w = x2 - x1
                if h > 0 and w > 0:
                    star_resized = cv2.resize(self.star_img, (w, h))
                    img[y1:y2, x1:x2] = star_resized
            except:
                self._draw_star_shape(img, x, y)
        else:
            self._draw_star_shape(img, x, y)

    def _draw_star_shape(self, img, x: int, y: int):
        """Draw a simple star shape."""
        size = 15
        # Draw simple star with lines
        points = []
        for i in range(5):
            angle = i * 4 * np.pi / 5 - np.pi / 2
            px = int(x + size * np.cos(angle))
            py = int(y + size * np.sin(angle))
            points.append([px, py])

        points = np.array([points[0], points[2], points[4], points[1], points[3]], np.int32)
        cv2.fillPoly(img, [points], (0, 215, 255))  # Gold star (BGR)
        cv2.polylines(img, [points], True, (0, 0, 0), 2)

    def _draw_globe(self, img, x: int, y: int):
        """Draw a globe/circle marker for safe zones."""
        if self.assets_loaded:
            # Use actual globe image
            try:
                y1 = max(0, y - 20)
                y2 = min(img.shape[0], y + 20)
                x1 = max(0, x - 20)
                x2 = min(img.shape[1], x + 20)

                h = y2 - y1
                w = x2 - x1
                if h > 0 and w > 0:
                    # If we have the glob asset, use it
                    glob_img = cv2.imread(os.path.join(os.path.dirname(__file__), '../ludo/assets/glob.png'))
                    if glob_img is not None:
                        glob_resized = cv2.resize(glob_img, (w, h))
                        img[y1:y2, x1:x2] = glob_resized
                        return
            except:
                pass

        # Fallback: Draw a shield/globe shape
        radius = 16
        # Outer circle (shield)
        cv2.circle(img, (x, y), radius, (50, 200, 50), -1)  # Green
        cv2.circle(img, (x, y), radius, (0, 0, 0), 2)
        # Inner circle
        cv2.circle(img, (x, y), radius - 5, (100, 255, 100), -1)  # Light green
        # Add "S" for Safe
        cv2.putText(img, "S", (x - 7, y + 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _draw_status(self, img, state_info):
        """Draw game status."""
        winner = state_info.get('winner', None)

        if winner is not None:
            # Draw winner banner
            banner_height = 80
            banner_y = self.board_height - banner_height - 20

            msg = f"GAME OVER - Player {winner} WINS!"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]

            # Center the banner
            box_x = (self.board_width - text_size[0]) // 2 - 30

            # Draw banner with shadow
            cv2.rectangle(img, (box_x + 5, banner_y + 5),
                         (box_x + text_size[0] + 65, banner_y + banner_height + 5),
                         (0, 0, 0), -1)  # Shadow
            cv2.rectangle(img, (box_x, banner_y),
                         (box_x + text_size[0] + 60, banner_y + banner_height),
                         self.GOAL_COLOR, -1)
            cv2.rectangle(img, (box_x, banner_y),
                         (box_x + text_size[0] + 60, banner_y + banner_height),
                         (0, 0, 0), 4)

            # Draw text
            cv2.putText(img, msg, (box_x + 30, banner_y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)

    def _draw_legend_simple(self, img, x: int, y: int):
        """Draw comprehensive legend with improved font matching heading style."""
        legend_width = 800
        legend_height = 180

        cv2.rectangle(img, (x, y), (x + legend_width, y + legend_height), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), 3)

        # Title - use same font as heading (FONT_HERSHEY_DUPLEX) with matching size
        cv2.putText(img, "Legend:", (x + 15, y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # Column 1: Players
        col1_x = x + 15
        cv2.putText(img, "Players:", (col1_x, y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for i in range(min(self.num_players, 2)):
            y_pos = y + 90 + i * 40
            color = self.PLAYER_COLORS[i]

            cv2.circle(img, (col1_x + 20, y_pos), 12, color, -1)
            cv2.circle(img, (col1_x + 20, y_pos), 12, (0, 0, 0), 2)

            label = f"P{i}" + (" (Agent)" if i == 0 else " (Opponent)")
            cv2.putText(img, label, (col1_x + 45, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Column 2: Special Tiles
        col2_x = x + 250
        cv2.putText(img, "Special Tiles:", (col2_x, y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Globe (Safe Zone)
        y_pos = y + 90
        self._draw_globe(img, col2_x + 20, y_pos)
        cv2.putText(img, "SAFE ZONE (Globe)", (col2_x + 45, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Cannot be captured!", (col2_x + 45, y_pos + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Star
        y_pos = y + 130
        self._draw_star_shape(img, col2_x + 20, y_pos)
        cv2.putText(img, "BONUS (Star)", (col2_x + 45, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Safe position", (col2_x + 45, y_pos + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Column 3: Zones
        col3_x = x + 540
        cv2.putText(img, "Zones:", (col3_x, y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Goal
        y_pos = y + 90
        cv2.rectangle(img, (col3_x + 5, y_pos - 12), (col3_x + 35, y_pos + 12),
                     self.GOAL_COLOR, -1)
        cv2.rectangle(img, (col3_x + 5, y_pos - 12), (col3_x + 35, y_pos + 12),
                     (0, 0, 0), 2)
        cv2.putText(img, "GOAL - Win here!", (col3_x + 45, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Normal track
        y_pos = y + 130
        cv2.rectangle(img, (col3_x + 5, y_pos - 12), (col3_x + 35, y_pos + 12),
                     self.TRACK_COLOR, -1)
        cv2.rectangle(img, (col3_x + 5, y_pos - 12), (col3_x + 35, y_pos + 12),
                     (0, 0, 0), 2)
        cv2.putText(img, "DANGER - Can capture!", (col3_x + 45, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def close(self):
        """Close visualization window."""
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except:
            pass
