import numpy as np
from typing import List
from ..utils.state import State
from .board_analyser import LudoBoardAnalyser


class OrthogonalStateAbstractor:
    """
    Converts raw Ludo game to 31-dimensional orthogonal state vector.
    """

    def __init__(self) -> None:
        """
        Initialize the orthogonal state abstractor.
        """
        self.globes = LudoBoardAnalyser.GLOBES
        self.stars = LudoBoardAnalyser.STARS
        self.home = LudoBoardAnalyser.HOME
        self.goal = LudoBoardAnalyser.GOAL  # FIX: Add missing goal constant
        self.home_corridor = LudoBoardAnalyser.HOME_STRETCH_INDEXS

    def get_orthogonal_state(self, state: State) -> np.ndarray:
        """
        Convert State Object to 31-dimensional orthogonal state vector.

        Args:
            state (State): The state object from LudoEnv 
        
        Returns:
         numpy array of shape (31,) with the orthogonal State Vector 
         """
        features = []

        # Per-piece features (20 dimensions)
        features.extend(self._get_per_piece_features(state))
        # Global Features (11 dims)
        features.extend(self._get_global_features(state))
        return np.array(features, dtype=np.float32)

    def _get_per_piece_features(self, state: State) -> List[float]:
        """
        Extract 20 features (5 per piece × 4 pieces)

        Features per piece:
        1. Normalized Progress (0.0-1.0)
        2. Is Safe (0 or 1)
        3. In home Corridor (0 or 1)
        4. Threat Distance (0.0-1.0)
        5. Kill Opportunity (0 or 1)
        """
        features = []

        for piece_idx in range(4):
            piece_pos = state.player_pieces[piece_idx]

            # Feature 1. Normalized Progress 
            progress = piece_pos / 57.0 
            features.append(progress)

            # Feature 2: Is safe 
            # FIX: Correct list concatenation
            safe_positions = self.stars + self.globes + [self.home, LudoBoardAnalyser.START]
            is_safe = 1 if piece_pos in safe_positions else 0
            features.append(float(is_safe))

            # Feature 3: In home corridor
            in_corridor = 1 if piece_pos in self.home_corridor else 0 
            features.append(float(in_corridor))

            # Feature 4: Threat Distance
            # FIX: Properly handle nested enemy_pieces and only consider enemies BEHIND
            threat_distances = []
            for enemy_pieces_list in state.enemy_pieces:  # Iterate over 3 enemies
                for enemy_pos in enemy_pieces_list:  # Iterate over 4 pieces per enemy
                    # Skip enemies at home or goal
                    if enemy_pos == LudoBoardAnalyser.HOME or enemy_pos == LudoBoardAnalyser.GOAL:
                        continue
                    # Calculate distance (positive = enemy behind, negative = enemy ahead)
                    distance = LudoBoardAnalyser.get_circular_distance(piece_pos, enemy_pos)
                    # Only consider threats from behind (1-6 steps)
                    if 1 <= distance <= 6:
                        threat_distances.append(distance)
            
            if threat_distances:
                min_threat = min(threat_distances)
                threat_distance = min(1.0, min_threat / 6.0)
            else:
                threat_distance = 1.0  # Safe (no threats)
            features.append(threat_distance)

            # Feature 5. Kill opportunity 
            # FIX: Use correct attribute name
            can_kill = 1 if LudoBoardAnalyser.can_capture(
                piece_pos, state.dice_roll, state.enemy_pieces
            ) else 0
            features.append(float(can_kill))

        return features

    def _get_global_features(self, state: State) -> List[float]:
        """
        Extract 11 Global Features 

        21. Relative Progress (-1.0-1.0)
        22. Pieces in Yard (0.0-1.0)
        23. Pieces Scored (0.0-1.0)
        24. Enemy Scored (0.0-1.0)
        25. Max Kill Potential (0.0-1.0)
        26-31. Dice Roll (0,1)^6)
        """
        features = []

        # Feature 21. Relative Progress
        # FIX: Properly calculate average of nested enemy_pieces
        my_avg_pos = np.mean(state.player_pieces)
        
        # Flatten enemy pieces for average calculation
        all_enemy_positions = []
        for enemy_list in state.enemy_pieces:
            all_enemy_positions.extend(enemy_list)
        enemy_avg_pos = np.mean(all_enemy_positions) if all_enemy_positions else 0.0
        
        relative_progress = (my_avg_pos - enemy_avg_pos) / 57.0
        features.append(relative_progress)

        # Feature 22. Pieces in Yard
        pieces_in_yard = np.sum(np.array(state.player_pieces) == self.home)
        features.append(pieces_in_yard / 4.0)

        # Feature 23. Pieces Scored 
        # FIX: Use correct goal constant
        pieces_scored = np.sum(np.array(state.player_pieces) == self.goal)
        features.append(pieces_scored / 4.0)

        # Feature 24. Enemy Scored 
        # FIX: Properly count across nested enemy_pieces
        enemy_scored = 0
        for enemy_list in state.enemy_pieces:
            enemy_scored += np.sum(np.array(enemy_list) == self.goal)
        features.append(enemy_scored / 12.0)

        # Feature 25. Max Kill Potential
        kill_count = 0
        for piece_pos in state.player_pieces:
            if LudoBoardAnalyser.can_capture(piece_pos, state.dice_roll, state.enemy_pieces):
                kill_count += 1
        max_kill_potential = min(1.0, kill_count / 4.0)
        features.append(max_kill_potential)

        # Feature 26-31. Dice Roll 
        # One-hot encode the dice roll 
        dice_one_hot = [0.0] * 6
        dice_one_hot[state.dice_roll - 1] = 1.0
        features.extend(dice_one_hot)

        return features