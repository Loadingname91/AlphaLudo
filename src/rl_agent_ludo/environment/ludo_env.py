"""
Pillar 1: LudoEnv (Environment Abstraction Layer)

Hardware abstraction layer (HAL) between project and ludopy library.
Provides a clean, Gym-like step()/reset() interface.
"""

import numpy as np
import ludopy
from typing import Dict, List, Optional, Tuple, Any
from ..utils.state import State
from .reward_shaper import RewardShaper, create_reward_shaper

# Visualization imports (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    print("Warning: opencv-python (cv2) not installed. Visualization will be disabled.")


class LudoEnv:
    """
    Ludo environment wrapper around ludopy library.
    
    Provides Gym-like interface: reset(), step(action)
    Handles state abstraction, opponent agents, and reward shaping.
    """
    
    def __init__(
        self,
        reward_schema: str = 'sparse',
        opponent_agents: Optional[List] = None,
        opponent_schedule: Optional[Dict] = None,
        player_id: int = 0,
        seed: Optional[int] = None,
        render: bool = False
    ):
        """
        Initialize Ludo environment.
        
        Args:
            reward_schema: Reward shaping strategy ('sparse', 'dense', 'decoupled-ila')
            opponent_agents: List of opponent agent instances (default: Random agents)
            opponent_schedule: Curriculum learning schedule for opponents (optional)
            player_id: Player ID for the learning agent (default: 0)
            seed: Random seed for reproducibility (optional)
            render: Whether to render the board using OpenCV (default: False)
        """
        self.reward_schema = reward_schema
        self.reward_shaper = create_reward_shaper(reward_schema)
        self.opponent_agents = opponent_agents or []
        self.opponent_schedule = opponent_schedule or {}
        self.player_id = player_id
        self.seed = seed
        self.render = render
        
        # Check if visualization dependencies are available
        if self.render and not HAS_CV2:
            print("Warning: opencv-python (cv2) not available, disabling rendering")
            self.render = False
        
        # Player ID mapping (ludopy uses 0-3, we need to track which is our learning agent)
        self.player_id_map = {i: i for i in range(4)}
        
        # Internal game state
        self.game: Optional[ludopy.Game] = None
        self.current_player = 0
        self.episode_step = 0
        self.done = False
        
        # State tracking
        self.last_state: Optional[State] = None
        
        # Episode tracking (for visualization)
        self.current_episode = 0
        
        # Visualization setup
        if self.render:
            self.window_name = f"Ludo - Player {self.player_id}"
            self.render_scale = 2.0  # Scale factor for better visibility
    
    def set_episode(self, episode_number: int) -> None:
        """
        Set current episode number (for visualization).
        
        Args:
            episode_number: Current episode number
        """
        self.current_episode = episode_number
    
    def reset(self) -> State:
        """
        Reset environment and return initial state.
        
        Starting player rotates across episodes to reduce first-player bias:
        - Episode 0: Player 0 starts
        - Episode 1: Player 1 starts
        - Episode 2: Player 2 starts
        - Episode 3: Player 3 starts
        - Episode 4: Player 0 starts (cycles)
        
        Returns:
            Initial State object
        """
        # Initialize new game (ludopy only takes ghost_players parameter)
        # Use seed + episode_number to ensure each episode is different but reproducible
        if self.seed is not None:
            import numpy as np
            episode_seed = self.seed + self.current_episode
            np.random.seed(episode_seed)
            import random
            random.seed(episode_seed)
        else:
            # No seed - use fully random (not reproducible)
            pass
        
        self.game = ludopy.Game(ghost_players=[])
        self.game.reset()  # Reset game state
        
        # Rotate starting player to reduce first-player bias
        # Episode 0: player 0 starts, Episode 1: player 1 starts, etc.
        starting_player = self.current_episode % 4
        self.game.current_player = starting_player
        
        self.episode_step = 0
        self.done = False
        
        # Get initial observation - ludopy requires get_observation() first
        obs, self.current_player = self.game.get_observation()
        dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = obs
        
        # Store current observation for state building
        self.current_obs = obs
        self.current_dice = dice
        self.current_move_pieces = list(move_pieces) if len(move_pieces) > 0 else []
        
        # Get initial state
        state = self._get_observation()
        self.last_state = state
        
        # Render initial state if enabled
        if self.render:
            self._render()
        
        return state
    
    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Execute one step in the environment - ONE game turn.
        
        Executes action for the current player (learning agent if it's their turn,
        opponent automatically if it's their turn).
        
        Args:
            action: Action index (piece to move). Only used if it's learning agent's turn.
                   Ignored for opponent turns (they play automatically).
        
        Returns:
            Tuple of (next_state, reward, done, info):
                - next_state: Next State object (for whoever's turn it is now)
                - reward: Reward value (float) - only non-zero on learning agent's actions
                - done: Whether episode is finished (bool)
                - info: Additional information dictionary with 'current_player' indicating whose turn it is
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        game_events = {}
        
        # Track whose turn it was BEFORE executing action
        was_learning_agent_turn = (self.current_player == self.player_id)
        
        # Check if it's learning agent's turn or opponent's turn
        if was_learning_agent_turn:
            # Learning agent's turn - execute their action
            game_events.update(self._handle_learning_agent_turn(action))
        else:
            # Opponent's turn - play automatically (action parameter ignored)
            game_events.update(self._handle_opponent_turn())
        
        # After executing action, advance to next observation
        self._advance_to_next_observation()
        
        # Check if game ended
        done, winners_info = self._check_game_ended()
        self.done = done
        
        # Build info dictionary
        info = self._build_info_dict(game_events, winners_info)
        info['episode_step'] = self.episode_step
        info['current_player'] = self.current_player
        info['is_learning_agent_turn'] = (self.current_player == self.player_id)
        info['learning_agent_player_id'] = self.player_id  # Always include learning agent ID
        
        # Calculate reward (only for learning agent's actions, otherwise 0)
        if was_learning_agent_turn or done:
            # Reward for learning agent's action, or final reward if game ended
            reward, ila_components = self.reward_shaper.get_reward(game_events)
            info.update(ila_components)
        else:
            # No reward for opponent turns (learning agent doesn't learn from opponent actions)
            reward = 0.0
        
        # Increment step counter
        self.episode_step += 1
        
        # Get next state (always return valid state, even if done)
        next_state = self._get_observation()
        self.last_state = next_state
        
        # Render if enabled
        if self.render:
            self._render()
        
        return next_state, reward, self.done, info
    
    def _advance_to_next_observation(self) -> None:
        """
        Advance to the next observation after an action has been executed.
        
        Updates current_player and observation data.
        """
        try:
            obs, self.current_player = self.game.get_observation()
            
            # Update current observation data
            dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = obs
            self.current_obs = obs
            self.current_dice = dice
            self.current_move_pieces = list(move_pieces) if len(move_pieces) > 0 else []
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "pending" in error_msg:
                # Observation already pending - shouldn't happen, but handle gracefully
                print(f"Warning: Unexpected pending observation: {e}")
                self.done = True
            else:
                raise
    
    def _handle_learning_agent_turn(self, action: int) -> Dict:
        """
        Handle learning agent's turn - execute their action.
        
        Args:
            action: Action index from agent
        
        Returns:
            Dictionary of game events
        """
        valid_moves = self.current_move_pieces
        
        # Map action index to piece_id
        if len(valid_moves) == 0:
            piece_to_move = 0  # No valid moves, pass
        elif action < 0 or action >= len(valid_moves):
            # Invalid action index, use first valid move as fallback
            piece_to_move = valid_moves[0] if len(valid_moves) > 0 else 0
        else:
            piece_to_move = valid_moves[action]
        
        # Execute action (calls answer_observation which advances game)
        return self._execute_action(piece_to_move)
    
    def _handle_opponent_turn(self) -> Dict:
        """
        Handle opponent's turn - execute opponent's action.
        
        Returns:
            Dictionary of game events
        """
        opponent_idx = self.current_player
        valid_moves = self.current_move_pieces
        
        # Check if we have an agent for this opponent
        if (opponent_idx < len(self.opponent_agents) and 
            self.opponent_agents[opponent_idx] is not None):
            # Use opponent agent
            opponent_agent = self.opponent_agents[opponent_idx]
            opponent_state = self._get_observation()
            opponent_action = opponent_agent.act(opponent_state)
            
            # Map action to piece_id
            if len(valid_moves) > 0 and opponent_action < len(valid_moves):
                piece_to_move = valid_moves[opponent_action]
            elif len(valid_moves) > 0:
                piece_to_move = valid_moves[0]
            else:
                piece_to_move = 0
        else:
            # Random action for opponent (use numpy random for independence from global random)
            if len(valid_moves) > 0:
                import numpy as np
                piece_idx = np.random.randint(0, len(valid_moves))
                piece_to_move = valid_moves[piece_idx]
            else:
                piece_to_move = 0  # No valid moves, pass
        
        return self._execute_action(piece_to_move)
    
    
    def _check_game_ended(self) -> Tuple[bool, Dict]:
        """
        Check if the game has ended and return winner information.
        
        Returns:
            Tuple of (done: bool, winners_info: dict)
        """
        winners = self.game.get_winners_of_game()
        winners_info = {}
        
        if len(winners) > 0:
            winners_info['winners'] = winners
            winners_info['winner_player_id'] = winners[0] if len(winners) > 0 else None  # First winner
            winners_info['won'] = (self.player_id in winners)
            return True, winners_info
        
        return False, winners_info
    
    def _build_info_dict(self, game_events: Dict, winners_info: Dict) -> Dict:
        """
        Build info dictionary from game events and winner information.
        
        Args:
            game_events: Dictionary of game events
            winners_info: Dictionary with winner information
        
        Returns:
            Info dictionary
        """
        info = {}
        
        # Add winner information if available
        if winners_info:
            info.update(winners_info)
            if 'won' in winners_info:
                if winners_info['won']:
                    game_events['player_won'] = True
                else:
                    game_events['opponent_won'] = True
        
        return info
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid action indices for current player.
        
        Returns:
            List of valid action indices
        """
        return self._get_valid_actions()
    
    def _get_valid_actions(self) -> List[int]:
        """Internal method to get valid actions."""
        if self.game is None:
            return [0]  # Default pass action
        
        # Valid moves come from current observation
        # current_move_pieces is a list of piece IDs that can be moved
        valid_moves = self.current_move_pieces if hasattr(self, 'current_move_pieces') else []
        
        if len(valid_moves) == 0:
            # No valid moves, return [0] for pass/no-op
            return [0]
        
        # Return indices for the valid moves (0-based indexing into valid_moves list)
        # The action is an index into the valid_moves list, not the piece_id directly
        return list(range(len(valid_moves)))
    
    def _execute_action(self, piece_id: int) -> Dict:
        """
        Execute action (move piece) in ludopy.
        
        Args:
            piece_id: Piece ID to move (0-3)
        
        Returns:
            Dictionary of game events that occurred
        """
        game_events = {}
        game_events['dice_roll'] = self.current_dice
        
        try:
            # Answer observation with chosen piece
            # ludopy will handle the move and advance to next player if needed
            after_obs = self.game.answer_observation(piece_id)
            
            # Detect if piece was moved (simplified - just check if there was a valid move)
            if len(self.current_move_pieces) > 0:
                game_events['piece_moved'] = True
            
            # Note: More detailed event detection (capture, goal entry) would require
            # comparing board states before and after, which is complex.
            # For now, we'll rely on sparse reward which only cares about win/loss.
            # This can be enhanced in Phase 2 with dense rewards.
            
        except RuntimeError as e:
            # This can happen if:
            # - observation wasn't pending (shouldn't happen)
            # - piece wasn't valid (ludopy validates this)
            # We'll re-raise to handle in step()
            raise
        except Exception as e:
            # Handle other ludopy errors gracefully
            print(f"Warning: Error executing action {piece_id}: {e}")
            raise
        
        return game_events
    
    def _get_observation(self) -> State:
        """
        Get current observation as State object.
        
        Returns:
            State object with full_vector, abstract_state, valid_moves, dice_roll
        """
        if self.game is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        # Get game state from ludopy
        # pieces is list of 4 lists (one per player), each with 4 piece positions
        pieces, enemy_pieces = self.game.get_pieces(self.current_player)
        
        # Get dice roll from current observation
        dice_roll = self.current_dice if hasattr(self, 'current_dice') else 1
        
        # Create state abstractions
        full_vector = self._get_full_state_vector(pieces, enemy_pieces, dice_roll)
        abstract_state = self._get_abstract_state(pieces, enemy_pieces, dice_roll)
        valid_moves = self._get_valid_actions()
        
        return State(
            full_vector=full_vector,
            abstract_state=abstract_state,
            valid_moves=valid_moves,
            dice_roll=dice_roll
        )
    
    def _get_full_state_vector(self, player_pieces: List[int], enemy_pieces: List[List[int]], dice_roll: int) -> np.ndarray:
        """
        Create full state vector for neural networks.
        
        Args:
            player_pieces: List of 4 piece positions for current player
            enemy_pieces: List of 3 lists, each with 4 piece positions for enemies
            dice_roll: Current dice roll (1-6)
        
        Returns:
            NumPy array of continuous features
        """
        features = []
        
        # Add current player's piece positions (4 pieces)
        for piece_pos in player_pieces:
            # Normalize position: -1 (home) -> 0, 0-57 (board) -> position/57
            if piece_pos == -1:  # Home
                features.append(0.0)
            else:
                features.append(float(piece_pos) / 57.0)
        
        # Add enemy pieces (3 enemies × 4 pieces = 12 positions)
        for enemy_pieces_list in enemy_pieces:
            for piece_pos in enemy_pieces_list:
                if piece_pos == -1:  # Home
                    features.append(0.0)
                else:
                    features.append(float(piece_pos) / 57.0)
        
        # Add dice roll (normalized to [0, 1])
        features.append(float(dice_roll) / 6.0)
        
        # Add current player (one-hot encoded)
        current_player_onehot = [0.0] * 4
        current_player_onehot[self.current_player] = 1.0
        features.extend(current_player_onehot)
        
        # Add learning agent's turn indicator
        features.append(1.0 if self.current_player == self.player_id else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_abstract_state(self, player_pieces: List[int], enemy_pieces: List[List[int]], dice_roll: int) -> tuple:
        """
        Create abstract state for tabular methods (hashable tuple).
        
        Args:
            player_pieces: List of 4 piece positions for current player
            enemy_pieces: List of 3 lists, each with 4 piece positions for enemies
            dice_roll: Current dice roll (1-6)
        
        Returns:
            Hashable tuple representing discrete state
        """
        # Simplified abstract state - will be expanded in Phase 1
        # For now, create a basic discretized representation
        
        # Discretize piece positions to bins
        # -1 (home) -> -1, 0-57 -> discretize to bins
        def discretize_pos(pos: int, bins: int = 10) -> int:
            """Discretize position to bins."""
            if pos == -1:  # Home
                return -1
            elif pos == 0:  # Start
                return 0
            elif pos >= 57:  # Goal
                return bins + 1
            else:
                # Bin positions: 0-5 (bin 1), 6-11 (bin 2), etc.
                return min((pos // 6) + 1, bins)
        
        # Discretize current player's pieces
        player_discrete = tuple(discretize_pos(pos) for pos in sorted(player_pieces))
        
        # Discretize enemy pieces (simplified - just count pieces in each bin)
        enemy_discrete = []
        for enemy_pieces_list in enemy_pieces:
            enemy_positions = tuple(discretize_pos(pos) for pos in sorted(enemy_pieces_list))
            enemy_discrete.append(enemy_positions)
        
        # Create hashable state tuple
        state_tuple = (
            player_discrete,  # Current player's discretized pieces
            tuple(enemy_discrete),  # Enemies' discretized pieces
            dice_roll,  # Current dice roll
            self.current_player,  # Current player
        )
        
        return state_tuple
    
    def _render(self) -> None:
        """
        Render the current board state using OpenCV.
        
        Uses ludopy's render_environment() method and displays using OpenCV.
        """
        if self.game is None or not self.render or not HAS_CV2:
            return
        
        try:
            # Get board image from ludopy (returns BGR format)
            board_img = self.game.render_environment()
            
            # Resize for better visibility
            height, width = board_img.shape[:2]
            new_width = int(width * self.render_scale)
            new_height = int(height * self.render_scale)
            board_img_resized = cv2.resize(board_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Font settings (larger text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2  # Bigger font
            font_thickness = 3  # Thicker lines
            line_height = 40  # Space between lines
            padding = 10
            
            # Left side text: Steps, Episode count
            left_text_lines = [
                f"Step: {self.episode_step}",
                f"Episode: {self.current_episode}"
            ]
            
            # Right side text: Current player, Dice
            right_text_lines = [
                f"Current Player: {self.current_player}",
                f"Dice: {self.current_dice}"
            ]
            
            # Draw left side text (top-left)
            left_x = padding
            left_y = padding + line_height
            
            for i, text in enumerate(left_text_lines):
                y_pos = left_y + (i * line_height)
                
                # Get text size for background
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # Draw background rectangle
                cv2.rectangle(
                    board_img_resized,
                    (left_x - 5, y_pos - text_size[1] - 5),
                    (left_x + text_size[0] + 5, y_pos + 5),
                    (0, 0, 0),  # Black background
                    -1
                )
                
                # Draw text
                cv2.putText(
                    board_img_resized,
                    text,
                    (left_x, y_pos),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
            
            # Draw right side text (top-right)
            right_x = new_width - padding
            
            for i, text in enumerate(right_text_lines):
                y_pos = left_y + (i * line_height)
                
                # Get text size for background and right-align
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = right_x - text_size[0]
                
                # Draw background rectangle
                cv2.rectangle(
                    board_img_resized,
                    (text_x - 5, y_pos - text_size[1] - 5),
                    (text_x + text_size[0] + 5, y_pos + 5),
                    (0, 0, 0),  # Black background
                    -1
                )
                
                # Draw text
                cv2.putText(
                    board_img_resized,
                    text,
                    (text_x, y_pos),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
            
            # Display image
            cv2.imshow(self.window_name, board_img_resized)
            cv2.waitKey(1)  # Non-blocking wait (1ms) - allows window to update
        
        except Exception as e:
            # If rendering fails, just continue without visualization
            print(f"Warning: Rendering failed: {e}")
    
    def close(self) -> None:
        """
        Close visualization windows and clean up resources.
        """
        if self.render and HAS_CV2:
            cv2.destroyAllWindows()
