"""
Rule-Based Heuristic Agent for Ludo.

Implements hand-crafted priority rules with phase-aware contextual multipliers.
"""

from typing import Optional, Dict, Any, List, Tuple
import random
import numpy as np

from .baseAgent import Agent
from ..utils.state import State


HOME_INDEX = 0
GOAL_INDEX = 57
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]  # Star positions
GLOBE_INDEXES = [1, 9, 22, 35, 48]  # Safe globe positions
HOME_CORRIDOR = list(range(52, 57))  # Positions 52-56

# Priority scores
WIN_MOVE = 1_000_000
SAFE_CAPTURE = 60_000  # Capture + Safe
CAPTURE_MOVE = 50_000
FLEE_MOVE = 50_000
HOME_BASE_PROGRESS = 10_000
BLOCKADE_CLUSTER = 4_000
FORM_BLOCKADE_MOVE = 3_500
GET_OUT_OF_HOME = 7_000

# Strategic bonuses
PROGRESS_SCORE = 1_000
STAR_JUMP = 5_500
GLOBE_HOP = 500
BALANCED_FRONT = 350
SPLIT_BLOCKADE = 250
# Slightly softer safe-wait bonus so we don't over-value sitting on globes in 4p games
SAFE_WAIT = 800   # Penalty magnitude for leaving safety without clear gain
ENDGAME_STACK = 1_500  # Bonus for keeping pieces together in endgame

# Risk penalties
RISK_FACTOR = 800.0
BLOCKADE_BREAK_PENALTY = 500
# Soften spawn camping penalty so we still exit home in crowded boards
SPAWN_CAMP_PENALTY = 7_000  # Penalty for exiting home into obvious danger


class RuleBasedAgent(Agent):
    """
    Rule baed heuristic agent for Ludo using priority rules and strategic bonuses.
    """
    def __init__(self, seed: Optional[int] = None,debug_scores: bool = False) -> None:
        """
        Initialize rule based agent 

        Args:
            seed: Random seed for tie-breaking 
            debug_scores: Print debug scores for each move
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.debug_scores = debug_scores
        self._last_score_debug : Optional[Dict[str, Any]] = None

    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        return False 
    
    def _calculate_score(self, state:State, piece_idx:int, action_idx:int) -> Tuple[float, str]:
        """
        Calculate the score for a given action using priority rules and strategic bonuses.

        Args:
            state: Current state of the game
            piece_idx: Index of the piece to move
            action_idx: Index of the action to take
        
        Returns:
            Tuple containing the score and the name of the rule that gave the score
        """
        current_pos = state.player_pieces[piece_idx]
        dice_roll = state.dice_roll

        # simulate move 
        next_pos = self._simulate_move(current_pos, dice_roll)

        # check priority rules and highest score wins
        game_phase = self._get_game_phase(state)

        #1. Win Move 
        if next_pos == GOAL_INDEX or self._is_at_goal(current_pos, dice_roll):
            multiplier = self._get_contextual_multiplier(game_phase, 'WIN_MOVE')
            return WIN_MOVE * multiplier, 'WIN_MOVE'

        # 2. Capture Move 
        if self._can_capture(next_pos, state.enemy_pieces):
            multiplier = self._get_contextual_multiplier(game_phase, 'CAPTURE_MOVE')
            if self._is_safe_position(next_pos):
                 # Safe capture is worth more
                 return SAFE_CAPTURE * multiplier, 'SAFE_CAPTURE'
            return CAPTURE_MOVE * multiplier, 'CAPTURE_MOVE'
        
        # 3. Flee Move 
        # Check if we can flee to safety (either safe position or significantly reducing threat)
        if self._is_threatened(current_pos, state.enemy_pieces):
            if self._is_safe_position(next_pos):
                # Can flee to absolute safety
                multiplier = self._get_contextual_multiplier(game_phase, 'FLEE_MOVE')
                return FLEE_MOVE * multiplier, 'FLEE_MOVE'
            elif self._significantly_reduces_threat(current_pos, next_pos, state.enemy_pieces):
                # Can flee to significantly safer position (reduces threat by moving away)
                multiplier = self._get_contextual_multiplier(game_phase, 'FLEE_MOVE')
                return FLEE_MOVE * multiplier * 0.7, 'FLEE_MOVE'  # Slightly lower than absolute safety
        
        # 4. Home Base Progress 
        if next_pos in HOME_CORRIDOR or next_pos == GOAL_INDEX:
            multiplier = self._get_contextual_multiplier(game_phase, 'HOME_BASE_PROGRESS')
            return HOME_BASE_PROGRESS * multiplier, 'HOME_BASE_PROGRESS'
        
        # 5. Blockade Cluster 
        if self._can_block_opponent_cluster(next_pos, state.enemy_pieces):
            multiplier = self._get_contextual_multiplier(game_phase, 'BLOCKADE_CLUSTER')
            return BLOCKADE_CLUSTER * multiplier, 'BLOCKADE_CLUSTER'
        
        # 6. Form Blockade Move
        if self._can_form_blockade(next_pos, state.player_pieces, piece_idx):
            return FORM_BLOCKADE_MOVE, 'FORM_BLOCKADE_MOVE'
        
        # 7. Get out of Home 
        if current_pos == HOME_INDEX and dice_roll == 6:
            # Check for spawn camping
            start_pos = 1
            if self._is_spawn_camped(start_pos, state.enemy_pieces):
                # If camped, apply penalty unless we can capture the camper
                 # Assuming we can't capture here because _can_capture check is above for next_pos=1
                return GET_OUT_OF_HOME - SPAWN_CAMP_PENALTY, 'GET_OUT_OF_HOME_CAMPED'

            # Bonus if multiple pieces are stuck at home
            pieces_at_home = sum(1 for p in state.player_pieces if p == HOME_INDEX)
            home_bonus = 1.0 + (pieces_at_home - 1) * 0.3  # Extra bonus for each additional stuck piece
            multiplier = self._get_contextual_multiplier(game_phase, 'GET_OUT_OF_HOME') * home_bonus
            return GET_OUT_OF_HOME * multiplier, 'GET_OUT_OF_HOME'
        
        # strategic bonuses
        score = 0.0
        rule_name = "STRATEGIC_BONUS"

        # Safe Wait Logic: If currently safe, penalize moving unless big reward
        if self._is_safe_position(current_pos) and not self._is_safe_position(next_pos):
            # Only if not making significant progress or capturing
             if not (next_pos in HOME_CORRIDOR or next_pos == GOAL_INDEX):
                  score -= SAFE_WAIT # Effectively a penalty for leaving safety
        
        # Endgame Stacking: Bonus for keeping pieces near each other
        if game_phase in ['closing', 'endgame'] and next_pos > 0:
             if self._is_near_friendly_piece(next_pos, state.player_pieces, piece_idx):
                  score += ENDGAME_STACK

        # Check for star jump once (used in multiple places)
        is_star_jump = self._is_star_jump(current_pos, next_pos, dice_roll)
        
        # Progress - reward based on actual distance traveled
        if current_pos == HOME_INDEX and next_pos > 0:
            # Exiting home is valuable
            score += PROGRESS_SCORE * 2
        elif next_pos > current_pos:
            # Reward actual distance traveled (more distance = more progress)
            distance = next_pos - current_pos
            # Account for star jumps (they provide extra distance)
            if is_star_jump:
                # Star jump already gives bonus, but also reward the distance
                score += PROGRESS_SCORE + (distance * 50)
            else:
                score += PROGRESS_SCORE + (distance * 20)
        elif current_pos in HOME_CORRIDOR and next_pos > current_pos:
            # Progress in home corridor is very valuable
            score += PROGRESS_SCORE * 3
        
        # Star jump (boost) - research shows this is more valuable when trailing
        if is_star_jump:
            if game_phase == 'trailing':
                score += STAR_JUMP * 1.2  # Research: Boost × 1.2 in trailing mode
            else:
                score += STAR_JUMP
        
        # Globe Hop
        if next_pos in GLOBE_INDEXES:
            score += GLOBE_HOP
        
        # Bonus for pieces close to goal (in home corridor)
        if next_pos in HOME_CORRIDOR:
            # Closer to goal = higher bonus
            distance_to_goal = GOAL_INDEX - next_pos
            score += (6 - distance_to_goal) * 200  # 52->1000, 56->200
        
        # Balanced Front (move least advanced piece)
        if self._is_least_advanced(piece_idx, state.player_pieces):
            score += BALANCED_FRONT
        
        # Risk penalty 
        risk = self._calculate_probability_of_risk(next_pos, state.enemy_pieces)
        
        # In trailing mode, still down-weight risk but not as aggressively (to avoid suiciding in 4p)
        if game_phase == 'trailing':
            score -= risk * 0.4  # Ignore 60% of risk instead of 90%
        else:
            score -= risk

        # Blockade handling
        is_in_blockade = self._is_in_blockade(piece_idx, state.player_pieces)
        if is_in_blockade and next_pos != current_pos:
            # Check if breaking blockade is strategic (moving to better position)
            if self._is_safe_position(next_pos) or next_pos in HOME_CORRIDOR:
                # Strategic split - moving to safety or goal path
                score += SPLIT_BLOCKADE
            else:
                # Breaking blockade without benefit
                score -= BLOCKADE_BREAK_PENALTY

        # If threatened but not fleeing, apply heavy penalty (but don't completely override)
        if self._is_threatened(current_pos, state.enemy_pieces) and rule_name != 'FLEE_MOVE':
            # Apply heavy penalty but keep other bonuses
            score -= risk * 2  # Double the risk penalty for staying in danger

        return score, rule_name

    def _simulate_move(self, current_pos: int, dice_roll: int) -> int:
        """
        Simulate a move on the board and return the new position.
        
        Handles:
        - Home exit (requires dice=6)
        - Star jumps (automatic forward jump)
        - Goal bounce (overshoot bounces back)
        - Home corridor movement
        
        Args:
            current_pos: Current position of the piece
            dice_roll: Dice roll to move
            
        Returns:
            New position after move
        """
        # Handle home position
        if current_pos == HOME_INDEX:
            if dice_roll == 6:
                return 6  # Exit home to start position
            return HOME_INDEX  # Stay home if not 6
        
        # Handle goal position (already finished)
        if current_pos == GOAL_INDEX:
            return GOAL_INDEX
        
        # Calculate next position
        next_pos = current_pos + dice_roll
        
        # Handle star jump (only on main board 1-51)
        if 1 <= current_pos <= 51 and next_pos in STAR_INDEXES:
            # Find next star in sequence
            star_idx = STAR_INDEXES.index(next_pos)
            if star_idx < len(STAR_INDEXES) - 1:
                # Jump to next star
                next_pos = STAR_INDEXES[star_idx + 1]
            else:
                # Last star, jump to goal
                next_pos = GOAL_INDEX
        
        # Handle goal bounce (overshoot from home corridor 52-56)
        # If we're in home corridor and overshoot goal, bounce back
        if current_pos in HOME_CORRIDOR and next_pos > GOAL_INDEX:
            overshoot = next_pos - GOAL_INDEX
            next_pos = GOAL_INDEX - overshoot
            # Ensure we stay in home corridor (52-56) or at goal
            if next_pos < HOME_CORRIDOR[0]:
                next_pos = HOME_CORRIDOR[0]
            elif next_pos > GOAL_INDEX:
                next_pos = GOAL_INDEX
        
        return next_pos
    
    def _is_at_goal(self, current_pos: int, dice_roll: int) -> bool:
        """
        Check if the piece reaches or overshoots the goal after the move.
        
        Args:
            current_pos: Current position of the piece
            dice_roll: Dice roll to move
            
        Returns:
            True if move reaches or overshoots goal
        """
        if current_pos == HOME_INDEX:
            return False
        
        # If already in home corridor, check if move reaches goal
        if current_pos in HOME_CORRIDOR:
            return current_pos + dice_roll >= GOAL_INDEX
        
        # For main board positions, goal is at 57
        if 1 <= current_pos <= 51:
            # Calculate if we reach goal (considering star jumps)
            next_pos = current_pos + dice_roll
            # If we land on or pass the last star (51), we can reach goal
            return next_pos >= GOAL_INDEX
        
        return False
    
    def _can_capture(self, next_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """
        Check if the move can capture an enemy piece.
        
        Args:
            next_pos: New position of the piece
            enemy_pieces: List of enemy piece positions
            
        Returns:
            True if move can capture an enemy piece
        """
        # Cannot capture on safe positions
        if self._is_safe_position(next_pos):
            return False
        
        # Check all enemy pieces
        for enemy in enemy_pieces:
            for enemy_pos in enemy:
                # Must be exact match and not at home/goal
                if enemy_pos == next_pos and enemy_pos not in [HOME_INDEX, GOAL_INDEX]:
                    return True
        return False

    def _is_threatened(self, current_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """
        Check if the piece is threatened by an enemy piece.
        
        Args:
            current_pos: Current position of the piece
            enemy_pieces: List of enemy piece positions
        """
        if self._is_safe_position(current_pos):
            return False
        for enemy in enemy_pieces:
            for enemy_pos in enemy:
                if enemy_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                dist = self._get_circular_distance(current_pos, enemy_pos)
                if 1<= dist <= 6:
                    return True
        return False
    
    def _is_safe_position(self, pos: int) -> bool:
        """
        Check if the position is safe (cannot be captured).
        
        Args:
            pos: Position to check
        """
        return (
            pos == HOME_INDEX or 
            pos == GOAL_INDEX or 
            pos in HOME_CORRIDOR or 
            pos in GLOBE_INDEXES or
            pos == 1  # Start position (safe globe)
        )

    def _get_circular_distance(self, pos1: int, pos2: int) -> int:
        """
        Calculate the circular distance between two positions on the main board.
        
        The main board is circular from positions 1-51. This calculates the minimum
        distance an enemy would need to travel to reach pos1 from pos2.
        
        Args:
            pos1: First position (target position)
            pos2: Second position (enemy position)
            
        Returns:
            Circular distance (1-25 for positions on main board), or 100 for invalid
        """
        # Handle special positions
        if pos1 == HOME_INDEX or pos2 == HOME_INDEX:
            return 100
        
        if pos1 == GOAL_INDEX or pos2 == GOAL_INDEX:
            return 100
        
        # Both positions must be on main board (1-51) for circular calculation
        if not (1 <= pos1 <= 51) or not (1 <= pos2 <= 51):
            return 100
        
        # Calculate circular distance using modulo arithmetic
        # The board wraps: distance from 51 to 1 is 1 (not 50)
        diff = abs(pos1 - pos2)
        # Since board is circular with 51 positions, wrap distance is 51 - diff
        circular_diff = min(diff, 51 - diff)
        
        return circular_diff
    
    def _can_block_opponent_cluster(self, next_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """
        Check if the move can block an opponent cluster.
        
        Args:
            next_pos: New position of the piece
            enemy_pieces: List of enemy piece positions
        """
        if self._is_safe_position(next_pos):
            return False
        
        for enemy in enemy_pieces:
            # Count pieces near this position
            nearby_count = 0
            for enemy_pos in enemy:
                if enemy_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                dist = self._get_circular_distance(next_pos, enemy_pos)
                if dist <= 6:
                    nearby_count += 1
            if nearby_count >= 2:
                return True
        return False
    
    def _can_form_blockade(self, next_pos: int, player_pieces: List[int], piece_idx: int) -> bool:
        """
        Check if the move can form a blockade.
        
        Args:
            next_pos: New position of the piece
            player_pieces: List of player piece positions
            piece_idx: Index of the piece to move
        """
        if next_pos in [HOME_INDEX, GOAL_INDEX]:
            return False
        
        count = sum(1 for i, pos in enumerate(player_pieces) 
                   if pos == next_pos and i != piece_idx)
        return count >= 1
    
    def _is_in_blockade(self, piece_idx: int, player_pieces: List[int]) -> bool:
        """
        Check if piece is in a blockade.
        
        Args:
            piece_idx: Index of the piece to check
            player_pieces: List of player piece positions
        """
        current_pos = player_pieces[piece_idx]
        if current_pos in [HOME_INDEX, GOAL_INDEX]:
            return False
        
        count = sum(1 for i, pos in enumerate(player_pieces) 
                   if pos == current_pos and i != piece_idx)
        return count >= 1
    
    def _is_star_jump(self, current_pos: int, next_pos: int, dice_roll: int) -> bool:
        """
        Check if move triggers a star jump (automatic forward movement).
        
        Star jumps occur when landing exactly on a star position, which
        automatically moves the piece to the next star or goal.
        
        Args:
            current_pos: Current position before move
            next_pos: Position after move (may have been adjusted by star jump)
            dice_roll: Dice roll used
            
        Returns:
            True if a star jump occurred
        """
        if current_pos == HOME_INDEX:
            return False
        
        # Only check star jumps on main board (1-51)
        if not (1 <= current_pos <= 51):
            return False
        
        # Calculate where we would land without star jump
        original_landing = current_pos + dice_roll
        
        # Check if we landed on a star and jumped forward
        if original_landing in STAR_INDEXES and next_pos != original_landing:
            return True
        
        return False

    def _is_least_advanced(self, piece_idx: int, player_pieces: List[int]) -> bool:
        """
        Check if piece is the least advanced (not at home).
        
        This encourages balanced piece advancement rather than leaving
        pieces behind.
        """
        current_pos = player_pieces[piece_idx]
        if current_pos == HOME_INDEX:
            return False
        
        # Find minimum position among pieces not at home
        pieces_on_board = [p for p in player_pieces if p != HOME_INDEX]
        if not pieces_on_board:
            return False
        
        min_pos = min(pieces_on_board)
        return current_pos == min_pos
    
    def _significantly_reduces_threat(
        self, 
        current_pos: int, 
        next_pos: int, 
        enemy_pieces: List[List[int]]
    ) -> bool:
        """
        Check if moving to next_pos significantly reduces threat compared to current_pos.
        
        Returns True if the threat level at next_pos is much lower than at current_pos.
        """
        if self._is_safe_position(next_pos):
            return True
        
        # Calculate threat levels
        current_threat = 0
        next_threat = 0
        
        for enemy in enemy_pieces:
            for enemy_pos in enemy:
                if enemy_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                
                # Current position threat
                dist_current = self._get_circular_distance(current_pos, enemy_pos)
                if 1 <= dist_current <= 6:
                    current_threat += (7 - dist_current)
                
                # Next position threat
                dist_next = self._get_circular_distance(next_pos, enemy_pos)
                if 1 <= dist_next <= 6:
                    next_threat += (7 - dist_next)
        
        # Significant reduction means threat is reduced by at least 50%
        if current_threat > 0:
            reduction_ratio = (current_threat - next_threat) / current_threat
            return reduction_ratio >= 0.5
        
        return False
    
    def _is_spawn_camped(self, start_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """Check if an enemy is camping the spawn point (index 1)."""
        for enemy in enemy_pieces:
            for enemy_pos in enemy:
                if enemy_pos in [HOME_INDEX, GOAL_INDEX]: continue
                # Enemy is at or just past spawn (1-6) relative to our start
                # Assuming global index 1 is our start.
                # Note: This needs careful mapping if coordinates are relative.
                # Assuming standard Ludopy where everyone starts at their own '1' or mapped.
                # Since we use 1-52 circular, let's assume start_pos 1 is global '1'.
                # A camper would be behind us or on us.
                # Simplified: check circular distance TO start pos
                dist = self._get_circular_distance(start_pos, enemy_pos)
                # If enemy is 1-6 steps BEHIND start_pos (i.e. dist from start to enemy is ~46-51)
                # OR if enemy is sitting on start_pos waiting
                
                # If enemy is at pos 1
                if enemy_pos == start_pos: return True
                
                # If enemy is waiting to kill (dist from enemy to start is 1-6)
                dist_to_spawn = self._get_circular_distance(start_pos, enemy_pos) 
                if 1 <= dist_to_spawn <= 6:
                     return True
        return False

    def _is_near_friendly_piece(self, pos: int, player_pieces: List[int], my_idx: int) -> bool:
        """Check if position is within 6 steps of another friendly piece."""
        for i, p_pos in enumerate(player_pieces):
            if i == my_idx: continue
            if p_pos in [HOME_INDEX, GOAL_INDEX]: continue
            dist = abs(pos - p_pos)
            if dist <= 6: return True
        return False

    def _calculate_probability_of_risk(self, next_pos: int, enemy_pieces: List[List[int]]) -> float:
        """
        Calculate probabilistic risk score for landing at a position.
        
        Risk increases as enemies get closer (1-6 steps away). The formula
        is: (7 - distance) * RISK_FACTOR, so closer enemies contribute more.
        
        Args:
            next_pos: Position to evaluate risk for
            enemy_pieces: List of enemy piece positions
            
        Returns:
            Risk score (higher = more dangerous)
        """
        # Safe positions have no risk
        if self._is_safe_position(next_pos):
            return 0.0
        
        risk_score = 0.0
        
        # Check all enemy pieces
        for enemy in enemy_pieces:
            for enemy_pos in enemy:
                # Skip enemies at home or goal (can't threaten)
                if enemy_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                
                # Calculate circular distance on main board
                dist = self._get_circular_distance(next_pos, enemy_pos)
                
                # Only consider enemies 1-6 steps away (can reach in one turn)
                if 1 <= dist <= 6:
                    # Closer enemies are more dangerous
                    # Distance 1: (7-1)*800 = 4800
                    # Distance 6: (7-6)*800 = 800
                    risk_score += (7 - dist) * RISK_FACTOR
        
        return risk_score

    def _get_weighted_equity_score(self, pieces: List[int]) -> float:
        """
        Calculate weighted equity score for a set of pieces.
        
        Based on research: Score = (Goal × 100) + (Home_Corridor × 50) + 
        (Safe_Globe × 10) + (Raw_Distance)
        
        Args:
            pieces: List of piece positions
            
        Returns:
            Weighted equity score
        """
        score = 0.0
        for pos in pieces:
            if pos == GOAL_INDEX:
                score += 100  # Goal position
            elif pos in HOME_CORRIDOR:
                score += 50 + pos  # Home corridor: 50 + position (52-56)
            elif pos in GLOBE_INDEXES or pos == 1:
                score += 10 + pos  # Safe globe: 10 + position
            elif pos == HOME_INDEX:
                score += 0  # Home: no progress
            else:
                score += pos  # Raw distance on main board
        return score
    
    def _get_game_phase(self, state: State) -> str:
        """
        Determine current game phase using weighted equity scoring.
        
        Uses research-based weighted equity scores to determine context:
        - Trailing: gap < -20 (panic mode)
        - Neutral: -20 <= gap <= 20 (balanced)
        - Leading: gap > 20 (lockdown mode)
        
        Also considers piece-based phases:
        - Opening: 2+ pieces at home
        - Endgame: 2+ pieces at goal
        - Closing: 2+ pieces in corridor/finished
        - Critical: Enemy has 2+ pieces at goal
        
        Args:
            state: Current state of the game
            
        Returns:
            Game phase string: 'opening', 'endgame', 'closing', 'critical', 
            'trailing', 'neutral', 'leading', or 'midgame'
        """
        player_pieces = state.player_pieces

        # Opening Phase: 2+ pieces at home (takes priority)
        pieces_at_home = sum(1 for p in player_pieces if p == HOME_INDEX)
        if pieces_at_home >= 2:
            return 'opening'
        
        # Endgame Phase: 2+ pieces at goal (player is winning)
        pieces_finished = sum(1 for p in player_pieces if p == GOAL_INDEX)
        if pieces_finished >= 2:
            return 'endgame'
        
        # Closing Phase: 2+ pieces in goal path or finished
        pieces_in_corridor = sum(1 for p in player_pieces if p in HOME_CORRIDOR)
        if pieces_in_corridor + pieces_finished >= 2:
            return 'closing'
        
        # Critical Phase: Enemy has 2+ pieces at goal (player is losing)
        for enemy in state.enemy_pieces:
            enemy_pieces_finished = sum(1 for p in enemy if p == GOAL_INDEX)
            if enemy_pieces_finished >= 2:
                return 'critical'
        
        # Context-based phase using weighted equity scores (research-based)
        my_score = self._get_weighted_equity_score(player_pieces)
        max_opponent_score = max(
            self._get_weighted_equity_score(enemy) 
            for enemy in state.enemy_pieces
        )
        gap = my_score - max_opponent_score
        
        if gap < -20:
            return 'trailing'  # Panic mode: high risk tolerance
        elif gap > 20:
            return 'leading'  # Lockdown mode: zero risk
        else:
            return 'neutral'  # Balanced race
        
    
    def _get_contextual_multiplier(self, game_phase: str, rule_name: str) -> float:
        """
        Get contextual multiplier for a given game phase and rule name.
        
        Based on research: Context-aware reward scaling where the same action
        has different strategic value depending on whether we're winning or losing.
        
        Research multipliers:
        - Trailing: Kill × 1.5, Boost × 1.2, Safety × 0.5, Risk × 0.5
        - Leading: Kill × 0.8, Safety × 2.0, Risk × 2.0
        - Neutral: All × 1.0 (baseline)
        
        Args:
            game_phase: Current game phase
            rule_name: Name of the rule/action
            
        Returns:
            Multiplier for the rule in this phase
        """
        multipliers = {
            'opening': {
                'GET_OUT_OF_HOME': 5.0,
                'WIN_MOVE': 2.0,
                'FLEE_MOVE': 1.5,
            },
            'closing': {
                'HOME_BASE_PROGRESS': 5.0,
                'WIN_MOVE': 2.0,
                'CAPTURE_MOVE': 1.5,
                'BLOCKADE_CLUSTER': 2.0,
                'FLEE_MOVE': 1.5,
            },
            'critical': {
                'CAPTURE_MOVE': 1.5,
                'BLOCKADE_CLUSTER': 2.0,
                'WIN_MOVE': 2.0,
                'FLEE_MOVE': 1.5,
            },
            'trailing': {
                # Research-based: Panic mode - high risk tolerance
                'CAPTURE_MOVE': 1.5,  # Value kills highly
                'WIN_MOVE': 1.0,  # Still important but not over-prioritized
                'FLEE_MOVE': 0.5,  # Low priority - accept risk
                'HOME_BASE_PROGRESS': 0.5,  # Less important when losing
                'BLOCKADE_CLUSTER': 1.2,  # Moderate blocking
            },
            'leading': {
                # Research-based: Lockdown mode - zero risk
                'CAPTURE_MOVE': 0.8,  # Less important when winning
                'WIN_MOVE': 1.0,  # Standard priority
                'FLEE_MOVE': 2.0,  # High priority - avoid risk
                'HOME_BASE_PROGRESS': 2.0,  # Strongly prioritize safety
                'BLOCKADE_CLUSTER': 1.5,  # Moderate blocking
            },
            'neutral': {
                # Balanced race - standard multipliers
                'WIN_MOVE': 2.0,
                'CAPTURE_MOVE': 1.5,
                'BLOCKADE_CLUSTER': 2.0,
                'FLEE_MOVE': 1.5,
                'HOME_BASE_PROGRESS': 1.0,
            },
            'midgame': {
                # Fallback to neutral behavior
                'WIN_MOVE': 2.0,
                'CAPTURE_MOVE': 1.5,
                'BLOCKADE_CLUSTER': 2.0,
                'FLEE_MOVE': 1.5,
            },
            'endgame': {
                'WIN_MOVE': 3.0,  # Strongly prioritize finishing
                'HOME_BASE_PROGRESS': 3.0,  # Get remaining pieces to goal
                'CAPTURE_MOVE': 1.5,
                'BLOCKADE_CLUSTER': 1.5,
                'FLEE_MOVE': 1.5,
            },
        }
        
        return multipliers.get(game_phase, {}).get(rule_name, 1.0)

    def get_last_score_debug(self) -> Optional[Dict[str, Any]]:
        """Get last move's score debug info.
        
        Returns:
            Last move's score debug info
        """
        return self._last_score_debug

    def act(self, state: State) -> int:
        """
        Select an action based on the state using priority rules and strategic bonuses.

        Args:
            state: Current state of the game
        
        Returns:
            Action index to take
        """
        if len(state.valid_moves) == 0:
            return 0 # no moves possible, return dummy action

        # shuffle for tie-breaking
        candidate_actions = list(state.valid_moves)
        random.shuffle(candidate_actions)

        best_action = candidate_actions[0]
        best_score = float('-inf')

        score_debug = []

        for action_idx in candidate_actions:
            # map action to piece index 
            if state.movable_pieces and len(state.movable_pieces) > action_idx:
                piece_idx = state.movable_pieces[action_idx]
            else:
                piece_idx = action_idx

            score, rule_name = self._calculate_score(state, piece_idx, action_idx)
            _score_debug = {
                'piece_idx': piece_idx,
                'action_idx': action_idx,
                'rule_name': rule_name,
                'score': score
            }
            score_debug.append(_score_debug)

            if score > best_score:
                best_score = score
                best_action = action_idx

        if self.debug_scores:
            self._last_score_debug = {
                'best_action': best_action,
                'best_score': best_score,
                'all_scores': score_debug
            }

        return best_action