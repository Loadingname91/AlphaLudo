import pytest
from rl_agent_ludo.utils.board_analyser import LudoBoardAnalyser

@pytest.fixture
def board_analyser():
    return LudoBoardAnalyser()

def test_is_at_home(board_analyser):
    assert board_analyser.is_at_home(0) == True
    assert board_analyser.is_at_home(1) == False
    assert board_analyser.is_at_home(2) == False
    assert board_analyser.is_at_home(3) == False
    assert board_analyser.is_at_home(4) == False
    assert board_analyser.is_at_home(5) == False


def test_is_at_goal(board_analyser):
    assert board_analyser.is_at_goal(57) == True
    assert board_analyser.is_at_goal(52) == False
    assert board_analyser.is_at_goal(53) == False


def test_is_on_goal_path(board_analyser):
    assert board_analyser.is_on_goal_path(51) == False
    assert board_analyser.is_on_goal_path(52) == True
    assert board_analyser.is_on_goal_path(53) == True
    assert board_analyser.is_on_goal_path(54) == True
    assert board_analyser.is_on_goal_path(55) == True
    assert board_analyser.is_on_goal_path(56) == True
    assert board_analyser.is_on_goal_path(57) == False


def test_is_on_glob(board_analyser):
    assert board_analyser.is_on_globe(9) == True
    assert board_analyser.is_on_globe(2) == False
    assert board_analyser.is_on_globe(3) == False
    assert board_analyser.is_on_globe(4) == False


def test_is_on_star(board_analyser):
    assert board_analyser.is_on_star(51) == True
    assert board_analyser.is_on_star(52) == False
    assert board_analyser.is_on_star(53) == False


def test_is_safe_position(board_analyser):
    assert board_analyser.is_safe_position(0) == True
    assert board_analyser.is_safe_position(1) == True
    assert board_analyser.is_safe_position(9) == True
    assert board_analyser.is_safe_position(51) == False
    assert board_analyser.is_safe_position(57) == True
    assert board_analyser.is_safe_position(58) == False
    assert board_analyser.is_safe_position(59) == False
    assert board_analyser.is_safe_position(32) == False


def test_is_threatened(board_analyser):
    assert board_analyser.is_threatened(1, [[2,3,44,43],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) == False
    assert board_analyser.is_threatened(2, [[1,3,44,43],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) == True
    assert board_analyser.is_threatened(3, [[1,2,44,43],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) == True


def test_can_capture(board_analyser):
    # Note: can_capture uses ludopy's get_enemy_at_pos which handles coordinate conversion
    # The coordinate conversion is complex and depends on ludopy's internal representation
    # Testing with raw positions may not work correctly due to coordinate system differences
    # These tests verify the safe zone logic which we can test independently
    
    # Test: Cannot capture in safe zones (globes)
    # Position 9 is a globe (safe zone), so cannot capture there
    # Even if enemy is at position 9, we can't capture because it's a safe zone
    assert board_analyser.can_capture(8, 1, [[9,10,11,12],[13,14,15,16],[17,18,19,20]]) == False
    
    # Test: Cannot capture at goal
    # Position 57 is goal (safe zone)
    assert board_analyser.can_capture(56, 1, [[57,0,0,0],[0,0,0,0],[0,0,0,0]]) == False
    
    # Test: Cannot capture at home
    # Position 0 is home (safe zone)
    assert board_analyser.can_capture(57, 1, [[0,0,0,0],[0,0,0,0],[0,0,0,0]]) == False
    
    # Note: Testing actual capture requires proper ludopy game state with correct coordinate conversion
    # The get_enemy_at_pos function needs enemy pieces in ludopy's coordinate system format 

