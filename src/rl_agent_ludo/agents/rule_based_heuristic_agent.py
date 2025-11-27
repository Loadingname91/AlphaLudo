"""
Rule-based heuristic agent.

Uses hand-crafted rules and scoring functions to evaluate moves.
Implements phase-aware contextual multipliers for different game situations.
"""
import random
from typing import Optional, List
from .base_agent import Agent
from ..utils.board_analyser import LudoBoardAnalyser
from ..utils.state import State

class RuleBasedHeuristicAgent(Agent):

    # Priority scores for different move types
    WIN_MOVE = 1000000  # Winning the game
    CAPTURE_MOVE = 50000  # Capturing an opponent
    FLEE_MOVE = 50000  # Fleeing from an opponent
    HOME_BASE_PROGRESS = 10000  # Moving in safe zones / home stretch
    BLOCKADE_CLUSTER = 4000  # Blocking an opponent's cluster
    FORM_BLOCKADE_MOVE = 3500  # Forming a blockade
    GET_OUT_OF_HOME = 7000  # Getting out of the home base

    # normal moves
    PROGRESS_SCORE = 1000
    GLOBE_HOP = 500
    STAR_JUMP = 5500
    BALANCED_FRONT = 350 # moving the piece furtthest behind
    SPLIT_BLOCKADE = 250 

    # --- 3. RISK PENALTY (The "Fear") ---
    # If a move lands in a dangerous spot, we SUBTRACT this score.
    # We scale it by distance: Closer enemy = Higher penalty.
    BLOCKADE_BREAK_PENALTY = 500
    RISK_FACTOR = 800.0

    # Game phase identifiers
    PHASE_OPENING = "OPENING"
    PHASE_MIDGAME = "MIDGAME"
    PHASE_CLOSING = "CLOSING"
    PHASE_CRITICAL = "CRITICAL"

    def __init__(self, seed: Optional[int] = None, debug_scores: bool = False):
        """Initialize rule-based heuristic agent."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Debugging configuration
        # When enabled, the agent will store a per-move score breakdown that can be
        # consumed by the trainer/metrics tracker via get_last_score_debug().
        self.debug_scores = debug_scores
        self._last_score_debug: Optional[dict] = None

    def _calculate_probablistic_risk(self, land_pos: int, enemy_pieces: List[List[int]]) -> float:
        """
        Calculate danger of landing at a position.
        
        Formula: Sum of (7 - distance) * RISK_FACTOR for all threatening enemies.
        Closer enemies contribute more to the risk score.
        """
        # Safe positions (globes, home stretch) have zero risk
        if LudoBoardAnalyser.is_safe_position(land_pos):
            return 0.0

        risk_score = 0.0

        # Check all enemy pieces
        for opponent_pieces in enemy_pieces:
            for enemy_pos in opponent_pieces:
                if enemy_pos == LudoBoardAnalyser.HOME or enemy_pos == LudoBoardAnalyser.GOAL:
                    continue

                dist = LudoBoardAnalyser.get_circular_distance(land_pos, enemy_pos)

                # Enemies 1-6 steps away are threatening
                # Distance 1: (7-1)*800 = 4800 penalty
                # Distance 6: (7-6)*800 = 800 penalty
                if 1 <= dist <= 6:
                    risk_score += (7 - dist) * self.RISK_FACTOR
        return risk_score

    @staticmethod
    def can_break_blockade(
        current_pos: int,
        next_pos: int,
        piece_idx: int,
        player_pieces: List[int],
    ) -> bool:
        """
        True if the current position is a blockade and the move breaks it.
        """
        # 1. Is this piece currently part of a blockade?
        in_blockade = LudoBoardAnalyser.piece_is_in_blockade(piece_idx, player_pieces)
        if not in_blockade:
            return False
        
        # 2. Will moving to a different position break the stack?
        # A move breaks the blockade if the piece moves AND the target is not the same piece's location.
        return next_pos != current_pos
        
    def _get_game_phase(self,player_pieces:List[int],enemy_pieces:List[List[int]]) -> str:
        """
        Determine the current game phase.
        """
        pieces_at_home = LudoBoardAnalyser.count_pieces_at_home(player_pieces)
        pieces_on_goal_path = LudoBoardAnalyser.count_pieces_on_goal_path(player_pieces)
        pieces_finished = LudoBoardAnalyser.count_pieces_at_goal(player_pieces)

        # Critical phase: enemy is close to winning (2+ pieces in goal)
        for opp in enemy_pieces:
            if LudoBoardAnalyser.count_pieces_at_goal(opp) >= 2:
                return self.PHASE_CRITICAL
        
        # Opening: 2+ pieces stuck at home, priority is getting them out
        if pieces_at_home >= 2:
            return self.PHASE_OPENING
        
        # Closing: pieces in home stretch or finished, focus on pushing to goal
        if pieces_on_goal_path + pieces_finished >= 2:
            return self.PHASE_CLOSING
        
        # Default: midgame standard play
        return self.PHASE_MIDGAME

    
    def _get_contextual_multipler(self, game_phase: str, rule_name: str) -> float:
        """Returns multiplier to boost specific rules based on game phase."""
        # Opening: prioritize getting pieces out of home
        if game_phase == self.PHASE_OPENING:
            if rule_name == 'GET_OUT_OF_HOME':
                return 5.0  # 7000 -> 35000 (almost as high as capture)
        
        # Closing: focus on finishing pieces
        if game_phase == self.PHASE_CLOSING:
            if rule_name == 'HOME_BASE_PROGRESS':
                return 5.0  # 10000 -> 50000 (equal to capture/flee)
            if rule_name == 'WIN_MOVE':
                return 2.0  # ensure nothing overrides winning
        
        # Critical: stop the leader aggressively
        if game_phase == self.PHASE_CRITICAL:
            if rule_name == 'CAPTURE_MOVE':
                return 1.5  # 50000 -> 75000 (prioritize killing)
            if rule_name == 'BLOCKADE_CLUSTER':
                return 2.0  # block aggressively
        
        return 1.0

    
    def _calculate_move_score(self, state: State, piece_idx: int, dice_roll: int) -> float:
        """
        Calculates the 'Goodness' of a move
        Layer 1 only checks the priority (Instincts)
        Layer 2 strategy
        """
        current_pos = state.player_pieces[piece_idx]
        next_pos = LudoBoardAnalyser.simulate_move(current_pos, dice_roll)
        
        # Prepare per-component breakdown for debugging
        components = {
            'win_move': 0.0,
            'capture_move': 0.0,
            'flee_move': 0.0,
            'home_base_progress': 0.0,
            'blockade_cluster': 0.0,
            'form_blockade_move': 0.0,
            'get_out_of_home': 0.0,
            'progress_score': 0.0,
            'star_jump': 0.0,
            'globe_hop': 0.0,
            'balanced_front': 0.0,
            'risk_penalty': 0.0,
            'blockade_break_penalty': 0.0,
        }

        # 1. Identify the Priority Rule Triggered
        base_score = 0.0
        rule_name = "NORMAL"

        if LudoBoardAnalyser.is_at_goal(next_pos):
            base_score = float(self.WIN_MOVE)
            rule_name = 'WIN_MOVE'
        elif LudoBoardAnalyser.can_capture(current_pos, dice_roll, state.enemy_pieces):
            base_score = float(self.CAPTURE_MOVE)
            rule_name = 'CAPTURE_MOVE'
        elif LudoBoardAnalyser.is_threatened(current_pos, state.enemy_pieces) and \
             (LudoBoardAnalyser.is_safe_position(next_pos) or not LudoBoardAnalyser.is_threatened(next_pos, state.enemy_pieces)):
            base_score = float(self.FLEE_MOVE)
            rule_name = 'FLEE_MOVE'
        elif LudoBoardAnalyser.is_on_goal_path(current_pos):
            base_score = float(self.HOME_BASE_PROGRESS)
            rule_name = 'HOME_BASE_PROGRESS'
        elif LudoBoardAnalyser.enemy_blockade_at(next_pos, state.enemy_pieces):
            base_score = float(self.BLOCKADE_CLUSTER)
            rule_name = 'BLOCKADE_CLUSTER'
        elif LudoBoardAnalyser.can_form_blockade(next_pos, state.player_pieces, piece_idx):
            base_score = float(self.FORM_BLOCKADE_MOVE)
            rule_name = 'FORM_BLOCKADE_MOVE'
        elif current_pos == LudoBoardAnalyser.HOME and dice_roll == LudoBoardAnalyser.DICE_MOVE_OUT_OF_HOME:
            base_score = float(self.GET_OUT_OF_HOME)
            rule_name = 'GET_OUT_OF_HOME'
        else:
            # Strategic Layer (Additive)
            base_score += self.PROGRESS_SCORE
            components['progress_score'] = float(self.PROGRESS_SCORE)
            
            if LudoBoardAnalyser.is_on_star(next_pos):
                base_score += self.STAR_JUMP
                components['star_jump'] = float(self.STAR_JUMP)
            
            if LudoBoardAnalyser.is_on_globe(next_pos):
                base_score += self.GLOBE_HOP
                components['globe_hop'] = float(self.GLOBE_HOP)
            
            least_advanced = LudoBoardAnalyser.get_least_advanced_piece_idx(state.player_pieces)
            if piece_idx == least_advanced:
                base_score += self.BALANCED_FRONT
                components['balanced_front'] = float(self.BALANCED_FRONT)
            
            if LudoBoardAnalyser.piece_is_in_blockade(piece_idx, state.player_pieces):
                base_score += self.SPLIT_BLOCKADE
            
            if self.can_break_blockade(current_pos, next_pos, piece_idx, state.player_pieces):
                base_score -= self.BLOCKADE_BREAK_PENALTY
                components['blockade_break_penalty'] = float(self.BLOCKADE_BREAK_PENALTY)
            
            rule_name = 'STRATEGY'

        # Apply contextual multiplier based on game phase
        phase = self._get_game_phase(state.player_pieces, state.enemy_pieces)
        multiplier = self._get_contextual_multipler(phase, rule_name)
        adjusted_score = base_score * multiplier

        # Update components dictionary for priority rules (strategic components already set above)
        if rule_name == 'WIN_MOVE':
            components['win_move'] = float(adjusted_score)
        elif rule_name == 'CAPTURE_MOVE':
            components['capture_move'] = float(adjusted_score)
        elif rule_name == 'FLEE_MOVE':
            components['flee_move'] = float(adjusted_score)
        elif rule_name == 'HOME_BASE_PROGRESS':
            components['home_base_progress'] = float(adjusted_score)
        elif rule_name == 'BLOCKADE_CLUSTER':
            components['blockade_cluster'] = float(adjusted_score)
        elif rule_name == 'FORM_BLOCKADE_MOVE':
            components['form_blockade_move'] = float(adjusted_score)
        elif rule_name == 'GET_OUT_OF_HOME':
            components['get_out_of_home'] = float(adjusted_score)

        # 3. Apply Risk (Always subtract from final)
        risk = self._calculate_probablistic_risk(next_pos, state.enemy_pieces)
        components['risk_penalty'] = float(risk)

        # Handle special case: threatened but not fleeing (moving to/remaining in danger)
        if LudoBoardAnalyser.is_threatened(current_pos, state.enemy_pieces) and rule_name != 'FLEE_MOVE':
            # Staying in (or moving to) danger without benefit is bad.
            final_score = float(-risk)
        else:
            final_score = float(adjusted_score - risk)

        # Store debug info if enabled
        if self.debug_scores:
            self._last_score_debug = {
                'piece_idx': int(piece_idx),
                'current_pos': int(current_pos),
                'next_pos': int(next_pos),
                'dice_roll': int(dice_roll),
                'components': components,
                'total_score': float(final_score),
            }

        return final_score

    

    @property
    def is_on_policy(self) -> bool:
        """Random agent doesn't learn, so this is not applicable."""
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """Random agent doesn't learn."""
        return False

    # ---- Optional score debugging hooks ----------------------------------------

    @property
    def supports_score_debug(self) -> bool:
        """Indicate that this agent can provide score breakdowns when enabled."""
        return self.debug_scores

    def get_last_score_debug(self) -> Optional[dict]:
        """
        Return debug information for the last scored move, if available.

        The returned dictionary is JSON-serializable and intended to be passed
        through MetricsTracker for offline analysis.
        """
        return self._last_score_debug

    def act(self, state: State) -> int:
        """
        Select action by evaluating all valid moves.
        
        Simulates each possible move, scores it using heuristics, and selects the highest.
        """
        if not state.valid_moves:
            return 0

        best_action = 0
        best_score = -float('inf')

        # Shuffle moves to break ties (otherwise always picks first piece)
        candidate_indices = list(range(len(state.valid_moves)))
        random.shuffle(candidate_indices)

        for action_idx in candidate_indices:
            # Map action index to actual piece index
            if state.movable_pieces:
                piece_idx = state.movable_pieces[action_idx]
            else:
                piece_idx = state.valid_moves[action_idx]

            score = self._calculate_move_score(state, piece_idx, state.dice_roll)

            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action



    def learn_from_replay(self, *args, **kwargs) -> None:
        """Random agent doesn't learn."""
        pass
    
    def learn_from_rollout(self, rollout_buffer: list, *args, **kwargs) -> None:
        """Random agent doesn't learn."""
        pass
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float,
                             next_state: State, done: bool, **kwargs) -> None:
        """Random agent doesn't use replay buffer."""
        pass