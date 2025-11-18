## 5. Rebuilding the Rule-Based Agent from Scratch – Checklist

This document is a **practical, linear checklist** to recreate the rule-based agent in an empty project.

---

### 5.1 Minimal File Layout

Target structure (names are suggestions, not strict requirements):

```text
your_project/
  agents/
    rule_based_heuristic_agent.py
  env/
    ludo_env.py
  utils/
    board_analyzer.py
    state.py
  configs/
    rule_based_heuristic_config.yaml
  train.py
```

---

### 5.2 Step 1 – Implement `State` DTO

Create `utils/state.py`:

- Define an immutable dataclass `State` with:
  - `full_vector: np.ndarray`
  - `abstract_state: tuple`
  - `valid_moves: list[int]`
  - `dice_roll: int`
  - `movable_pieces: list[int] | None = None`
  - `player_pieces: list[int] | None = None`
  - `enemy_pieces: list[list[int]] | None = None`
- In `__post_init__`, assert:
  - `1 <= dice_roll <= 6`
  - `len(valid_moves) >= 1`
  - Types of `full_vector` and `abstract_state` are correct.

This gives you a stable interface between environment and agent.

---

### 5.3 Step 2 – Implement `LudoBoardAnalyzer`

Create `utils/board_analyzer.py`:

1. Define constants: `HOME`, `GOAL`, `STARS`, `GLOBES`, `DICE_TO_EXIT_HOME`.  
2. Implement predicates:
   - `is_at_home`, `is_at_goal`, `is_on_goal_path`, `is_on_star`, `is_on_globe`, `is_safe_position`.  
   - Blockade helpers: `piece_is_in_blockade`, `can_form_blockade`, `enemy_blockade_at`.  
   - Threat & capture: `is_threatened`, `can_capture`.  
3. Implement helpers:
   - `simulate_move`, `get_star_jump_target`, `distance_to_goal`, `distance_to_next_star`.  
   - `get_zone`, `count_pieces_at_home`, `count_pieces_on_goal_path`.  
   - `get_least_advanced_piece_idx`, `get_most_advanced_piece_idx`.  
   - `min_enemy_distance_behind_blockade`, `is_enemy_spawn_danger`.

You can copy the logic from the pseudocode in `02_board_analyzer_core_logic.md`.

---

### 5.4 Step 3 – Implement Environment Wrapper

Create `env/ludo_env.py` that:

- Wraps your Ludo engine (or a custom implementation).  
- On `reset()`:
  - Initializes the game.  
  - Gets initial `player_pieces`, `enemy_pieces`, `dice_roll`, `move_pieces`.  
  - Builds `full_vector` and `abstract_state`.  
  - Returns a `State` with all fields populated.
- On `step(action)`:
  - Maps `action` (index into `valid_moves`) to `piece_to_move`.  
  - Applies the move in the engine.  
  - Computes `game_events` for reward shaping if desired.  
  - Builds new `State`.  
  - Returns `(next_state, reward, done, info)`.

Use `04_environment_and_state_integration.md` as your reference for details.

---

### 5.5 Step 4 – Implement `RuleBasedHeuristicAgent`

Create `agents/rule_based_heuristic_agent.py` with:

1. **Configurable constants**:
   - Priority scores: `WIN_MOVE`, `CAPTURE_MOVE`, `FLEE_MOVE`, `HOME_STRETCH_PROGRESS`, `BLOCKADE_BUSTER`, `FORM_BLOCKADE`, `GET_OUT_OF_HOME`.  
   - Strategic scores: `PROGRESS_SCORE`, `GLOBE_HOP`, `SPLIT_BLOCKADE`, `BALANCED_FRONT`.  
   - Game phases: `"OPENING"`, `"MIDGAME"`, `"CLOSING"`, `"CRITICAL"`.  
   - Risk settings: `capture_penalty`, `enable_risk_calculation`, `risk_threshold`.

2. **Public API**:
   - `is_on_policy` → `False`.  
   - `needs_replay_learning` → `False`.  
   - `act(state: State) -> int`:
     - Reconstruct `player_pieces` / `enemy_pieces` if not provided.  
     - Optionally build cached data (`threats`, `blockades`, `game_phase`, `least_advanced_idx`).  
     - Loop over candidate actions and call `calculate_move_score`.  
     - Return the index of the best action.

3. **Private helpers**:
   - `calculate_move_score` (full pipeline).  
   - `calculate_priority_score`.  
   - `calculate_strategic_score`.  
   - `get_game_phase`.  
   - `get_contextual_multiplier`.  
   - `calculate_probabilistic_risk`.  
   - Small predicates: `_is_win_move`, `_is_get_out_of_home`, `_is_flee_move`, `_is_capture_move`, `_is_home_stretch_progress`, `_is_form_blockade`, `_is_star_jump`, `_is_near_star`, `_get_progress_score`, `_is_breaking_safe_blockade`.

Use `03_scoring_engine_core_logic.md` as the reference for these functions.

---

### 5.6 Step 5 – Wire a Simple Trainer / Runner

Create `train.py` (or equivalent) that:

```python
env = LudoEnv(...)
agent = RuleBasedHeuristicAgent(...)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        if env.current_player == env.player_id:
            action = agent.act(state)
        else:
            action = 0  # ignored or used for opponents
        next_state, reward, done, info = env.step(action)
        state = next_state
```

Since the agent is rule-based, you don’t need gradient updates or replay buffers – the trainer is mostly for **statistics and evaluation**.

---

### 5.7 Step 6 – Add Tests for Core Logic

At minimum, write tests that:

- Verify `LudoBoardAnalyzer` utilities:  
  - Stars, globes, goal path, move simulation, blockades, capture, threat.  
- Verify priority scoring:  
  - Winning moves dominate.  
  - Captures > regular moves.  
  - Fleeing from threats works.  
  - Home-stretch and blockade rules fire correctly.  
- Verify strategic scoring:  
  - Progress, star bonuses, blockades, balanced front, lead piece, hunter bonus.  
- Verify decision making:  
  - Agent prefers win moves.  
  - Agent prefers captures over pure progress when appropriate.  
  - Agent does not stall when moves are available.

With these pieces in place, you have everything needed to implement the **entire rule-based agent business logic from scratch**, independent of this codebase.


