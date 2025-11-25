from rl_agent_ludo.utils.orthogonal_state_abstractor import OrthogonalStateAbstractor
from rl_agent_ludo.utils.state import State
import numpy as np


def create_mock_state() -> State:
    """
    Create a mock state for testing.
    FIX: enemy_pieces must be List[List[int]] (3 enemies × 4 pieces)
    """
    return State(
        full_vector=np.zeros(31),  # Placeholder
        abstract_state=(0, 0, 0, 0, 0),  # Placeholder
        valid_moves=[0, 1, 2, 3],
        dice_roll=1,
        player_pieces=[0, 1, 2, 3],
        enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],  # FIX: Nested list
        movable_pieces=[0, 1, 2, 3]
    )


def test_orthogonal_abstractor():
    """
    Test that abstractor produces correct shape and ranges.
    """
    # FIX: Create abstractor instance
    abstractor = OrthogonalStateAbstractor()
    
    # Create a mock state 
    state = create_mock_state()

    # FIX: Use correct method name
    features = abstractor.get_orthogonal_state(state)

    # Check Shape
    assert features.shape == (31,), f"Expected 31 features, got {features.shape}"

    # Check feature level values 
    
    # Per Piece features (20 Dimensions) - Test first piece
    piece_idx = 0
    base_idx = piece_idx * 5
    
    # Normalized Progress [0.0,1.0]
    assert 0.0 <= features[base_idx] <= 1.0, f"Normalized Progress out of range: {features[base_idx]}"

    # Is safe {0,1}
    assert features[base_idx + 1] in [0, 1], f"Is safe should be 0 or 1, got {features[base_idx + 1]}"

    # In home corridor {0,1}
    assert features[base_idx + 2] in [0, 1], f"In home corridor should be 0 or 1, got {features[base_idx + 2]}"

    # Threat Distance [0.0,1.0]
    assert 0.0 <= features[base_idx + 3] <= 1.0, f'Threat Distance out of range: {features[base_idx + 3]}'

    # Kill Opportunity {0,1}
    assert features[base_idx + 4] in [0, 1], f"Kill Opportunity should be 0 or 1, got {features[base_idx + 4]}"

    # Global Features (11 Dimensions)
    
    # Relative Progress [-1.0,1.0]
    assert -1.0 <= features[20] <= 1.0, f"Relative Progress out of range: {features[20]}"

    # Pieces in Yard [0.0,1.0]
    assert 0.0 <= features[21] <= 1.0, f"Pieces in Yard out of range: {features[21]}"

    # Pieces Scored [0.0,1.0]
    assert 0.0 <= features[22] <= 1.0, f"Pieces Scored out of range: {features[22]}"

    # Enemy Scored [0.0,1.0]
    assert 0.0 <= features[23] <= 1.0, f"Enemy Scored out of range: {features[23]}"

    # Max Kill Potential [0.0,1.0]
    assert 0.0 <= features[24] <= 1.0, f"Max Kill Potential out of range: {features[24]}"
    
    # Dice Roll [0,1]^6
    assert len(features[25:]) == 6, f"Dice Roll should be 6 dimensions, got {len(features[25:])}"
    assert all(0.0 <= f <= 1.0 for f in features[25:]), f"Dice Roll values out of range: {features[25:]}"
    assert sum(features[25:]) == 1.0, f"Dice Roll values should sum to 1.0, got {sum(features[25:])}"

def test_orthogonal_abstractor_all_pieces_at_goal():
    """
    Test that abstractor produces correct values when all pieces are at goal.
    """
    # Create a mock state with all pieces at goal
    abstractor = OrthogonalStateAbstractor()
    
    # Test: All pieces at goal
    state2 = State(
        full_vector=np.zeros(31),
        abstract_state=(0, 0, 0, 0, 0),
        valid_moves=[0],  # Pass action when no moves available
        dice_roll=1,
        player_pieces=[57, 57, 57, 57],
        enemy_pieces=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        movable_pieces=[]
    )
    features2 = abstractor.get_orthogonal_state(state2)
    assert features2[22] == 1.0  # All pieces scored


def test_orthogonal_abstractor_all_pieces_at_home():
    """
    Test that abstractor produces correct values when all pieces are at home.
    """
    # Create a mock state with all pieces at home
    abstractor = OrthogonalStateAbstractor()
    
    # Test: All pieces at home
    state1 = State(
        full_vector=np.zeros(31),
        abstract_state=(0, 0, 0, 0, 0),
        valid_moves=[0],
        dice_roll=6,
        player_pieces=[0, 0, 0, 0],
        enemy_pieces=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        movable_pieces=[0]
    )
    features1 = abstractor.get_orthogonal_state(state1)
    assert features1.shape == (31,)
    assert features1[21] == 1.0  # All pieces in yard