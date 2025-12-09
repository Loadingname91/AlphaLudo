"""
Simple CV2 Visualizer for Curriculum Levels.
Creates graphical windows showing game state for all 5 levels.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class CurriculumVisualizer:
    """Visualizer for curriculum-based Ludo environments."""

    def __init__(self, level: int, num_players: int = 2, tokens_per_player: int = 1):
        """
        Initialize visualizer.

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
        self.width = 1200
        self.height = 800

        # Colors (BGR format for OpenCV)
        self.BG_COLOR = (240, 240, 240)
        self.TRACK_COLOR = (200, 200, 200)
        self.PLAYER_COLORS = [
            (100, 200, 100),  # Player 0 (Agent) - Green
            (100, 100, 200),  # Player 1 - Red
            (200, 100, 100),  # Player 2 - Blue
            (200, 200, 100),  # Player 3 - Yellow
        ]
        self.TEXT_COLOR = (50, 50, 50)
        self.GOAL_COLOR = (255, 215, 0)  # Gold
        self.SAFE_ZONE_COLOR = (100, 255, 100)  # Light green for safe zones
        self.CAPTURE_ZONE_COLOR = (255, 200, 200)  # Light red for capture zones

        # Track settings
        self.track_length = 60  # Standard Ludo track
        self.cell_size = 15
        # Safe zones (positions where pawns cannot be captured)
        self.safe_zones = [10, 20, 30, 40, 50] if level >= 3 else []

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
        # Create blank canvas
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        img[:] = self.BG_COLOR

        # Draw title
        self._draw_title(img, state_info)

        # Draw game info
        self._draw_game_info(img, state_info)

        # Draw board based on level
        if self.level <= 2:
            self._draw_simple_track(img, state_info)
        else:
            self._draw_multi_token_board(img, state_info)

        # Draw status
        self._draw_status(img, state_info)

        # Show window
        cv2.imshow(self.window_name, img)
        cv2.waitKey(100)  # 100ms delay

    def _draw_title(self, img, state_info):
        """Draw title bar."""
        title = f"Level {self.level} - Curriculum Learning"
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                   self.TEXT_COLOR, 2, cv2.LINE_AA)

    def _draw_game_info(self, img, state_info):
        """Draw game information."""
        y_offset = 80

        # Step counter
        step = state_info.get('step', 0)
        cv2.putText(img, f"Step: {step}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Current player
        current = state_info.get('current_player', 0)
        cv2.putText(img, f"Current Player: {current}", (20, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Dice (if applicable)
        if 'dice' in state_info and state_info['dice'] is not None:
            dice = state_info['dice']
            cv2.putText(img, f"Dice: {dice}", (20, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)

            # Draw dice visual
            self._draw_dice_visual(img, dice, (200, y_offset + 35))

    def _draw_dice_visual(self, img, dice: int, pos: Tuple[int, int]):
        """Draw a visual representation of the dice."""
        x, y = pos
        size = 40

        # Draw dice box
        cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 2)

        # Draw dots based on dice value
        dot_positions = {
            1: [(20, 20)],
            2: [(10, 10), (30, 30)],
            3: [(10, 10), (20, 20), (30, 30)],
            4: [(10, 10), (30, 10), (10, 30), (30, 30)],
            5: [(10, 10), (30, 10), (20, 20), (10, 30), (30, 30)],
            6: [(10, 10), (30, 10), (10, 20), (30, 20), (10, 30), (30, 30)],
        }

        for dx, dy in dot_positions.get(dice, []):
            cv2.circle(img, (x + dx, y + dy), 3, (0, 0, 0), -1)

    def _draw_simple_track(self, img, state_info):
        """Draw simple linear track (for Level 1-2)."""
        positions = state_info.get('player_positions', [[0], [0]])

        # Track position
        track_y = 250
        start_x = 100

        # Draw track background
        track_width = self.track_length * self.cell_size
        cv2.rectangle(img, (start_x, track_y - 30),
                     (start_x + track_width, track_y + 30),
                     self.TRACK_COLOR, -1)

        # Draw safe zones and capture zones if applicable
        if self.safe_zones:
            self._draw_zone_markers(img, start_x, track_y, track_width)

        # Draw goal
        goal_x = start_x + track_width - 20
        cv2.rectangle(img, (goal_x, track_y - 30),
                     (goal_x + 20, track_y + 30),
                     self.GOAL_COLOR, -1)
        cv2.putText(img, "GOAL", (goal_x - 10, track_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)

        # Draw position markers every 10 steps
        for i in range(0, self.track_length + 1, 10):
            x = start_x + i * self.cell_size
            cv2.line(img, (x, track_y - 35), (x, track_y - 40), (100, 100, 100), 1)
            cv2.putText(img, str(i), (x - 8, track_y - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Draw tokens for each player
        for player_idx, player_tokens in enumerate(positions):
            color = self.PLAYER_COLORS[player_idx]

            for token_idx, pos in enumerate(player_tokens):
                # Clamp position to track
                pos = max(0, min(pos, self.track_length))

                # Calculate pixel position
                x = start_x + int(pos * self.cell_size)

                # Offset multiple tokens at same position
                y_offset = (token_idx - len(player_tokens)/2 + 0.5) * 15

                # Draw token
                cv2.circle(img, (x, int(track_y + y_offset)), 12, color, -1)
                cv2.circle(img, (x, int(track_y + y_offset)), 12, (0, 0, 0), 2)

                # Draw player label
                cv2.putText(img, f"P{player_idx}", (x - 10, int(track_y + y_offset + 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw legend
        self._draw_legend(img, 100, 400)

    def _draw_multi_token_board(self, img, state_info):
        """Draw multi-token board (for Level 3-5)."""
        positions = state_info.get('player_positions', [])

        # Calculate number of tracks needed
        total_tokens = sum(len(p) for p in positions)

        # Draw each player's tokens on separate mini-tracks
        start_y = 200
        track_spacing = 120

        for player_idx, player_tokens in enumerate(positions):
            y_pos = start_y + player_idx * track_spacing

            # Player label
            color = self.PLAYER_COLORS[player_idx]
            label = f"Player {player_idx}" + (" (Agent)" if player_idx == 0 else "")
            cv2.putText(img, label, (20, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw track for this player
            start_x = 250
            track_width = self.track_length * self.cell_size

            cv2.rectangle(img, (start_x, y_pos - 15),
                         (start_x + track_width, y_pos + 15),
                         self.TRACK_COLOR, -1)

            # Draw safe zones and capture zones
            if self.safe_zones:
                self._draw_zone_markers(img, start_x, y_pos, track_width, track_height=30)

            # Draw goal
            goal_x = start_x + track_width - 20
            cv2.rectangle(img, (goal_x, y_pos - 15),
                         (goal_x + 20, y_pos + 15),
                         self.GOAL_COLOR, -1)

            # Draw tokens
            for token_idx, pos in enumerate(player_tokens):
                pos = max(-1, min(pos, self.track_length))

                if pos == -1:  # Token at home
                    x = start_x - 30
                elif pos >= self.track_length:  # Token at goal
                    x = goal_x + 10
                else:
                    x = start_x + int(pos * self.cell_size)

                # Draw token
                cv2.circle(img, (x, y_pos), 10, color, -1)
                cv2.circle(img, (x, y_pos), 10, (0, 0, 0), 2)

                # Token number
                cv2.putText(img, str(token_idx), (x - 4, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw legend for multi-token boards (positioned on the right side)
        self._draw_legend(img, 900, 200)

    def _draw_status(self, img, state_info):
        """Draw game status."""
        winner = state_info.get('winner', None)

        if winner is not None:
            # Game over - draw winner message
            msg = f"GAME OVER - Player {winner} WINS!"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]

            # Draw background box
            box_x = (self.width - text_size[0]) // 2 - 20
            box_y = self.height - 120
            cv2.rectangle(img, (box_x, box_y),
                         (box_x + text_size[0] + 40, box_y + 80),
                         self.GOAL_COLOR, -1)
            cv2.rectangle(img, (box_x, box_y),
                         (box_x + text_size[0] + 40, box_y + 80),
                         (0, 0, 0), 3)

            # Draw text
            cv2.putText(img, msg, (box_x + 20, box_y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)

    def _draw_zone_markers(self, img, start_x: int, track_y: int, track_width: int, track_height: int = 60):
        """Draw safe zones and capture zones on the track."""
        zone_width = self.cell_size  # Width of each zone marker
        
        # Draw safe zones (green markers)
        for safe_pos in self.safe_zones:
            if safe_pos < self.track_length:
                zone_x = start_x + int(safe_pos * self.cell_size)
                # Draw a green rectangle to mark safe zone
                cv2.rectangle(img, 
                             (zone_x - zone_width // 2, track_y - track_height // 2),
                             (zone_x + zone_width // 2, track_y + track_height // 2),
                             self.SAFE_ZONE_COLOR, -1)
                # Add border
                cv2.rectangle(img,
                             (zone_x - zone_width // 2, track_y - track_height // 2),
                             (zone_x + zone_width // 2, track_y + track_height // 2),
                             (0, 150, 0), 2)
        
        # Draw capture zones (light red markers between safe zones)
        # Capture zones are positions 1-59 excluding safe zones, home (0), and goal (60)
        capture_zones = [pos for pos in range(1, self.track_length) 
                         if pos not in self.safe_zones]
        
        # Draw subtle markers for capture zones (every 5 positions to avoid clutter)
        for capture_pos in capture_zones[::5]:  # Every 5th position
            zone_x = start_x + int(capture_pos * self.cell_size)
            # Draw a subtle red dot to indicate capture zone
            cv2.circle(img, (zone_x, track_y), 3, self.CAPTURE_ZONE_COLOR, -1)

    def _draw_legend(self, img, x: int, y: int):
        """Draw color legend with improved font matching heading style."""
        # Use same font as heading (FONT_HERSHEY_DUPLEX) with slightly smaller size
        legend_title = "Legend:"
        cv2.putText(img, legend_title, (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, self.TEXT_COLOR, 2, cv2.LINE_AA)

        y_offset = 35
        for i in range(self.num_players):
            y_pos = y + y_offset + i * 35
            color = self.PLAYER_COLORS[i]

            # Draw color box
            cv2.rectangle(img, (x, y_pos - 12), (x + 25, y_pos + 12), color, -1)
            cv2.rectangle(img, (x, y_pos - 12), (x + 25, y_pos + 12), (0, 0, 0), 2)

            # Draw label with matching font style
            label = f"Player {i}" + (" (Agent)" if i == 0 else " (Random)")
            cv2.putText(img, label, (x + 35, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Add safe zone and capture zone legend if applicable
        if self.safe_zones:
            y_pos = y + y_offset + self.num_players * 35 + 15
            
            # Safe zone legend
            cv2.rectangle(img, (x, y_pos - 12), (x + 25, y_pos + 12), 
                         self.SAFE_ZONE_COLOR, -1)
            cv2.rectangle(img, (x, y_pos - 12), (x + 25, y_pos + 12), (0, 150, 0), 2)
            cv2.putText(img, "Safe Zone (No Capture)", (x + 35, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)
            
            y_pos += 35
            
            # Capture zone legend
            cv2.circle(img, (x + 12, y_pos), 5, self.CAPTURE_ZONE_COLOR, -1)
            cv2.circle(img, (x + 12, y_pos), 5, (200, 0, 0), 1)
            cv2.putText(img, "Capture Zone (Vulnerable)", (x + 35, y_pos + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.TEXT_COLOR, 1, cv2.LINE_AA)

    def close(self):
        """Close the visualization window."""
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)  # Process window events
        except:
            pass
