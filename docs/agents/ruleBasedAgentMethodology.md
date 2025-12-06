# Rule-Based Heuristic Agent Methodology

## Theory

The Rule-Based Heuristic Agent uses hand-crafted priority rules with phase-aware contextual multipliers to evaluate moves. It implements a two-layer evaluation system: **Instincts** (priority rules) and **Strategy** (additive bonuses).

## Rule Hierarchy

Moves are evaluated in priority order (highest priority wins):

1. **WIN_MOVE** (1,000,000): Winning the game (reaching goal)
2. **CAPTURE_MOVE** (50,000): Capturing an opponent piece
3. **FLEE_MOVE** (50,000): Fleeing from a threatened position to safety
4. **HOME_BASE_PROGRESS** (10,000): Moving in safe zones / home stretch
5. **BLOCKADE_CLUSTER** (4,000): Blocking an opponent's cluster
6. **FORM_BLOCKADE_MOVE** (3,500): Forming a blockade (2+ pieces at same position)
7. **GET_OUT_OF_HOME** (7,000): Getting a piece out of home base
8. **STRATEGY** (additive bonuses): Progress, star jumps, globe hops, balanced front

## Game Phases

The agent adapts its behavior based on game phase:

### Opening Phase
- **Condition**: 2+ pieces stuck at home
- **Strategy**: Prioritize getting pieces out of home
- **Multiplier**: GET_OUT_OF_HOME × 5.0 (7,000 → 35,000)

### Midgame Phase
- **Condition**: Default phase (not opening, closing, or critical)
- **Strategy**: Standard play with balanced priorities

### Closing Phase
- **Condition**: 2+ pieces in home stretch or finished
- **Strategy**: Focus on finishing pieces
- **Multipliers**:
  - HOME_BASE_PROGRESS × 5.0 (10,000 → 50,000)
  - WIN_MOVE × 2.0 (ensure nothing overrides winning)

### Critical Phase
- **Condition**: Enemy has 2+ pieces in goal
- **Strategy**: Stop the leader aggressively
- **Multipliers**:
  - CAPTURE_MOVE × 1.5 (50,000 → 75,000)
  - BLOCKADE_CLUSTER × 2.0 (block aggressively)

## Scoring System

### Priority Scores (Base Values)

| Rule | Score | Description |
|------|-------|-------------|
| WIN_MOVE | 1,000,000 | Reaching goal |
| CAPTURE_MOVE | 50,000 | Capturing enemy |
| FLEE_MOVE | 50,000 | Escaping threat |
| HOME_BASE_PROGRESS | 10,000 | Moving in safe zones |
| BLOCKADE_CLUSTER | 4,000 | Blocking opponent |
| FORM_BLOCKADE_MOVE | 3,500 | Forming blockade |
| GET_OUT_OF_HOME | 7,000 | Exiting home |

### Strategic Bonuses (Additive)

| Bonus | Score | Description |
|-------|-------|-------------|
| PROGRESS_SCORE | 1,000 | Base progress reward |
| STAR_JUMP | 5,500 | Landing on star (jump bonus) |
| GLOBE_HOP | 500 | Landing on globe (safe) |
| BALANCED_FRONT | 350 | Moving least advanced piece |
| SPLIT_BLOCKADE | 250 | Breaking own blockade |

### Risk Penalties (Subtractive)

| Penalty | Formula | Description |
|---------|---------|-------------|
| RISK_FACTOR | 800.0 | Base risk multiplier |
| Risk Score | Sum of (7 - distance) × RISK_FACTOR | For enemies 1-6 steps behind |
| BLOCKADE_BREAK_PENALTY | 500 | Penalty for breaking own blockade |

## Risk Calculation

### Probabilistic Risk Formula

```python
risk_score = 0.0
for enemy in enemy_pieces:
    for enemy_pos in enemy:
        if enemy_pos in [HOME, GOAL]:
            continue
        dist = get_circular_distance(land_pos, enemy_pos)
        if 1 <= dist <= 6:
            risk_score += (7 - dist) * RISK_FACTOR
```

**Examples**:
- Enemy 1 step behind: (7-1) × 800 = 4,800 penalty
- Enemy 6 steps behind: (7-6) × 800 = 800 penalty

### Safe Position Handling
- If `is_safe_position(land_pos)` → return 0.0 (bypass risk calculation)
- Safe positions: Home, Start, Globes, Home Corridor, Goal

## Action Selection Algorithm

1. **Evaluate all valid moves**:
   - For each action in `state.valid_moves`:
     - Map action index to piece index (via `state.movable_pieces`)
     - Simulate move: `next_pos = simulate_move(current_pos, dice_roll)`
     - Calculate move score using `_calculate_move_score()`

2. **Score Calculation**:
   - Identify priority rule triggered (highest priority wins)
   - Apply contextual multiplier based on game phase
   - Add strategic bonuses (if no priority rule)
   - Subtract risk penalty

3. **Tie-breaking**:
   - Shuffle candidate moves before evaluation
   - Select move with highest score
   - If tie, random selection (due to shuffle)

## Edge Cases

### No Valid Moves
- **Scenario**: `state.valid_moves` is empty
- **Handling**: Return `0` as fallback
- **Rationale**: Should not happen, but provides graceful degradation

### Tie-Breaking
- **Scenario**: Multiple moves have equal scores
- **Handling**: Shuffle candidate indices before evaluation, then select first best
- **Rationale**: Prevents always selecting first piece, adds randomness

### Blockade Detection
- **Scenario**: 2+ pieces at same position
- **Handling**: 
  - Use `LudoBoardAnalyser.piece_is_in_blockade()` to check
  - Use `LudoBoardAnalyser.can_form_blockade()` to detect formation opportunity
- **Edge Case**: Blockade at home/goal (cannot form blockade there)

### Risk Calculation for Safe Positions
- **Scenario**: Move lands on safe position (globe, home corridor, etc.)
- **Handling**: Return 0.0 immediately (bypass risk calculation)
- **Rationale**: Safe positions cannot be captured, so no risk

### Phase Detection Boundaries
- **Opening**: `pieces_at_home >= 2` (exact boundary: 2 pieces)
- **Closing**: `pieces_on_goal_path + pieces_finished >= 2` (exact boundary: 2 pieces)
- **Critical**: `count_pieces_at_goal(opponent) >= 2` (exact boundary: 2 pieces)
- **Edge Cases**:
  - 0 pieces at home but 1 piece finished → Midgame (not Opening)
  - 1 piece in goal path, 1 finished → Closing (boundary case)
  - All pieces finished → Closing phase

### Invalid Piece Positions
- **Scenario**: Piece at home (0) or goal (57)
- **Handling**:
  - Home: Check if `dice_roll == 6` for GET_OUT_OF_HOME rule
  - Goal: Check `is_at_goal()` for WIN_MOVE rule
- **Rationale**: Special handling for terminal positions

### Blockade Break Detection
- **Scenario**: Piece is in blockade and move would break it
- **Handling**:
  1. Check if piece is in blockade: `piece_is_in_blockade(piece_idx, player_pieces)`
  2. Check if move breaks it: `next_pos != current_pos`
  3. Apply BLOCKADE_BREAK_PENALTY (subtract 500)
- **Rationale**: Breaking own blockade is usually bad (loses defensive advantage)

### Threatened but Not Fleeing
- **Scenario**: Piece is threatened, but move doesn't flee (rule_name != 'FLEE_MOVE')
- **Handling**: Set `final_score = -risk` (negative risk as penalty)
- **Rationale**: Staying in danger without benefit is worse than neutral

### Empty Enemy Pieces
- **Scenario**: `state.enemy_pieces` is empty or all enemies at home/goal
- **Handling**: Risk calculation returns 0.0 (no threats)
- **Rationale**: No enemies to threaten, so no risk

### Action Index Mapping
- **Scenario**: `state.movable_pieces` may be None or empty
- **Handling**: 
  - If `state.movable_pieces` exists: `piece_idx = state.movable_pieces[action_idx]`
  - Else: `piece_idx = state.valid_moves[action_idx]`
- **Rationale**: Handle different state representations

## Implementation Checklist

- [ ] Initialize with optional seed and debug_scores flag
- [ ] Implement game phase detection (`_get_game_phase`)
- [ ] Implement contextual multipliers (`_get_contextual_multipler`)
- [ ] Implement risk calculation (`_calculate_probablistic_risk`)
- [ ] Implement blockade break detection (`can_break_blockade`)
- [ ] Implement move scoring (`_calculate_move_score`)
- [ ] Handle all priority rules in order
- [ ] Apply strategic bonuses for non-priority moves
- [ ] Apply risk penalties
- [ ] Handle edge cases (empty moves, safe positions, etc.)
- [ ] Implement action selection with tie-breaking
- [ ] Set `is_on_policy = False` and `needs_replay_learning = False`
- [ ] Implement optional score debugging (`supports_score_debug`, `get_last_score_debug`)

