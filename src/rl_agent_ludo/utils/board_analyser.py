import enum
from pickle import FALSE
from token import STAR
from turtle import distance
from typing import List, Tuple

from ludopy.player import HOME_AREAL_INDEXS, get_enemy_at_pos, NO_ENEMY


# Constants from the board
HOME_INDEX = 0
START_INDEX = 1
GOAL_INDEX = 57
HOME_STRETCH_INDEXS = HOME_AREAL_INDEXS  # The safe "Home Stretch"
GLOB_INDEXS = [9, 22, 35, 48]             # Safe Globes
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51] # Stars (Jump points)
DICE_MOVE_OUT_OF_HOME = 6
ENEMY_GLOBS = [14,27,40] # #Enemy globs

class LudoBoardAnalyser:
    """
    Analyses the state of the board and return accordingly the information about the board.
    """

    HOME = HOME_INDEX
    START = START_INDEX
    GOAL = GOAL_INDEX
    GLOBES = GLOB_INDEXS
    STARS = STAR_INDEXS
    DICE_MOVE_OUT_OF_HOME = DICE_MOVE_OUT_OF_HOME
    HOME_STRETCH_INDEXS = HOME_STRETCH_INDEXS


    # Core Physics (Simulation and distance )

    @staticmethod
    def get_circular_distance(target_pos:int,source_pos:int)->int:
        """
        Calculates the distance on a 52 title main loop
        Handles wrap around so the agent sees the enemies behind it at the loop restart
        returns : Number of steps 'source' behind the target'
        """

        # if either is in special zones like Home , goal path , goal
        if target_pos>GOAL_INDEX or source_pos >GOAL_INDEX or source_pos == HOME_INDEX or target_pos == HOME_INDEX:
            return target_pos-source_pos 
        
        dist = target_pos-source_pos 
        if dist <0:
            dist +=52
        return dist

    # tile checking

    @staticmethod
    def simulate_move(current_pos:int,dice_roll:int)->int:
        """
        Predicts where as peice will land given a dice roll.
        Handles : star jumps , goal bounce and home exit
        """

        # case 0 stuck at home 
        if current_pos == HOME_INDEX:
            if dice_roll == DICE_MOVE_OUT_OF_HOME:
                return START_INDEX
            return HOME_INDEX
        
        # case 1 : already at goal 
        if current_pos == GOAL_INDEX:
            return GOAL_INDEX
        
        # case 2 : calculate raw new position 
        predicted_pos = current_pos + dice_roll

        # goal bounce since the dice gave more than what position is allowed 
        if predicted_pos > GOAL_INDEX:
            overshoot = predicted_pos - GOAL_INDEX
            predicted_pos = GOAL_INDEX - overshoot
        

        # case 3 : goal stretch entry ( from main board )
        # if player is on the main board and passed 51 (without jumping from 51)
        # note logic handled naturally by indicies 52-56 unless we wrapped 51->0
        # which isnt possible for player piece

        # case 4 : star jump 
        # if we land exactly on a star , we jump to the next one
        if predicted_pos in STAR_INDEXS:
            try:
                idx = STAR_INDEXS.index(predicted_pos)
                if predicted_pos == 51:
                    return GOAL_INDEX
                return STAR_INDEXS[idx+1]
            except IndexError:
                raise ValueError(f"Star jump index error: {predicted_pos} not in {STAR_INDEXS}")
                # should never happen since we check for 51 above
        
        return predicted_pos
        
    @staticmethod
    def is_at_home(pos:int)->bool:
        return pos == HOME_INDEX

    @staticmethod
    def is_at_goal(pos:int)->bool:
        return pos == GOAL_INDEX
    
    @staticmethod
    def is_on_goal_path(pos:int)->bool:
        return pos in HOME_STRETCH_INDEXS
    
    @staticmethod
    def is_on_globe(pos:int)->bool:
        return pos in GLOB_INDEXS

    @staticmethod
    def is_on_star(pos:int)->bool:
        return pos in STAR_INDEXS

    @staticmethod
    def is_safe_position(pos: int) -> bool:
        """Returns True if position is Home, Globe, Goal Path, or Goal."""
        return (pos == HOME_INDEX or 
                pos in GLOB_INDEXS or 
                pos in HOME_STRETCH_INDEXS or 
                pos == GOAL_INDEX or
                pos == START_INDEX) # Start is also safe (globe)

    # Game logic ( threats , blockades , captures)

    @staticmethod
    def is_threatened(piece_pos:int,enemy_pieces:List[List[int]])->bool:
        """
        check if any enemy is 1-6 steps behind this piece or near an enemy spawn
        Uses get_circular_distance to ensure no blind spots
        """
        if LudoBoardAnalyser.is_safe_position(piece_pos):
            return False

        for opponent_pieces in enemy_pieces:
            for enemy_pos in opponent_pieces:
                danger_zone = [GOAL_INDEX,HOME_INDEX]
                if enemy_pos in danger_zone:
                    continue 

                # check the distance from current 
                dist = LudoBoardAnalyser.get_circular_distance(piece_pos,enemy_pos)

                if 1 <= dist <=6 :
                    return True 
        if LudoBoardAnalyser.is_enemy_spawn_danger(piece_pos):
            return True 
        return False

    @staticmethod
    def can_form_blockade(new_pos: int, player_pieces: List[int], moving_piece_idx: int) -> bool:
        """
        Check if moving to new_pos forms a blockade.
        
        Args:
            new_pos: Target position
            player_pieces: List of 4 piece positions
            moving_piece_idx: Index of piece being moved (0-3)
        
        Returns:
            True if move forms a blockade
        """
        # Can't form blockade at home or goal
        if new_pos == HOME_INDEX or new_pos == GOAL_INDEX:
            return False
        
        # Check if any other piece is at new_pos
        for idx, pos in enumerate(player_pieces):
            if idx != moving_piece_idx and pos == new_pos:
                return True
        
        return False

    @staticmethod
    def can_capture(from_pos:int,dice_roll:int,enemy_pieces:list[list[int]])->bool:
        """
        Returns true if move lands on an enemy (sending them home).
        Uses ludopy's coordinate conversion to handle different enemy coordinate frames.
        """
        predicted_pos = LudoBoardAnalyser.simulate_move(from_pos,dice_roll)
        
        # Cannot capture in safe zones (goal, home, or globes)
        # Note: START_INDEX (1) is safe but you CAN capture enemies from there
        # You can capture on Stars if you land on them before the jump,
        # but ludopy simplifies this - capture usually happens at final destination
        safe_zones = [GOAL_INDEX, HOME_INDEX] + GLOB_INDEXS
        if predicted_pos in safe_zones:
            return False
        
        # Use ludopy's get_enemy_at_pos to check if any enemy is at predicted position
        # This handles coordinate conversion automatically
        enemy_at_pos, enemy_pieces_at_pos = get_enemy_at_pos(predicted_pos, enemy_pieces)
        
        # If enemy_at_pos is not NO_ENEMY, we can capture
        return enemy_at_pos != NO_ENEMY

    
    @staticmethod
    def find_blockades(player_pieces:List[int])-> List[Tuple[int,List[int]]]:
        """
        returns a list of (pos,[piece_indicies]) where 2+ pieces are stacked 
        """

        from collections import defaultdict
        counts = defaultdict(list)
        for idx,pos in enumerate(player_pieces):
            if pos!=HOME_INDEX and pos!=GOAL_INDEX:
                counts[pos].append(idx)
        
        return [(pos,idxs) for pos,idxs in counts.items() if len(idx)>=2]

    @staticmethod
    def piece_is_in_blockade(piece_idx:int,player_pieces:List[int])->bool:
        pos = player_pieces[piece_idx]
        if pos == HOME_INDEX or pos == GOAL_INDEX:
            return False
        
        count = sum(1 for p in player_pieces if p == pos)
        return count >=2 # since you need atleast 2 pieces to form a blockade


    @staticmethod
    def enemy_blockade_at(pos: int, enemy_pieces: List[List[int]]) -> bool:
        """True if enemies have a blockade at 'pos'."""
        for opponent_pieces in enemy_pieces:
            count = sum(1 for p in opponent_pieces if p == pos)
            if count >= 2:
                return True
        return False
    
    # strategic helpers

    @staticmethod
    def distance_to_goal(pos:int)->int:
        if pos==HOME_INDEX:return 57
        if pos>=GOAL_INDEX:return 0 
        return GOAL_INDEX-pos
    
    @staticmethod
    def get_least_advanced_piece_idx(player_pieces:List[int])->int:
        # Filter valid pieces (not home/goal)
        candidates = [(i, p) for i, p in enumerate(player_pieces) 
                     if p != HOME_INDEX and p != GOAL_INDEX]

        if not candidates:
            return 0

        # Return the index (i) of the piece furthest from the goal (highest distance to goal)
        return min(candidates, key=lambda x: x[1])[0]

    @staticmethod
    def is_enemy_spawn_danger(pos:int)->bool:
        """True if position is just ahead of an enemy spawn (risky). """
        for spawn in ENEMY_GLOBS:
            # check 1-6 titles ahead of spawn 
            if 1 <= (pos-spawn) <=6:
                return True 
        return False 


    @staticmethod
    def count_pieces_at_home(player_pieces:List[int])->int:
        return sum(1 for p in player_pieces if p==LudoBoardAnalyser.HOME)

    @staticmethod
    def count_pieces_at_goal(player_pieces:List[int])->int:
        return sum(1 for p in player_pieces if p==LudoBoardAnalyser.GOAL)

    @staticmethod
    def count_pieces_on_goal_path(player_pieces:List[int])->int:
        return sum(1 for p in player_pieces if p in HOME_STRETCH_INDEXS)