# Random Agent Methodology

## Theory

The Random Agent is a baseline validation agent that selects random valid actions. It does not learn and serves as a control to validate the environment, training loop, and metrics tracking systems.

## Purpose

- **Baseline Validation**: Establishes expected win rate of ~25% (1/4 players in a 4-player game)
- **Environment Testing**: Validates that the Ludo environment is functioning correctly
- **Infrastructure Validation**: Confirms that Trainer, LudoEnv, and MetricsTracker work properly

## Implementation

### Action Selection

The agent selects a random action from the list of valid moves:

```python
def act(self, state: State) -> int:
    if not state.valid_moves:
        return 0  # Fallback
    return random.choice(state.valid_moves)
```

### Learning Methods

All learning methods are no-ops (do nothing):
- `learn_from_replay()`: Pass (no-op)
- `learn_from_rollout()`: Pass (no-op)
- `push_to_replay_buffer()`: Pass (no-op)

### Agent Properties

- `is_on_policy`: Returns `False` (not applicable, doesn't learn)
- `needs_replay_learning`: Returns `False` (doesn't use replay buffer)

## Edge Cases

### Empty Valid Moves List
- **Scenario**: `state.valid_moves` is empty or falsy
- **Handling**: Return `0` as fallback action
- **Rationale**: Should not happen in normal gameplay, but provides graceful degradation

### Seed Handling
- **Scenario**: Seed parameter is `None` vs integer
- **Handling**: 
  - If `seed is not None`: Call `random.seed(seed)` for reproducibility
  - If `seed is None`: Use system random (non-deterministic)
- **Rationale**: Allows both deterministic (testing) and non-deterministic (baseline) operation

### No Learning Methods
- **Scenario**: Trainer calls learning methods on random agent
- **Handling**: All learning methods return `None` or `pass` (no-op)
- **Rationale**: Random agent doesn't maintain state or learn, so these methods are intentionally empty

## Expected Performance

- **Win Rate**: ~25% (1/4 players, assuming equal random chance)
- **Use Case**: Environment validation, baseline comparison

## Implementation Checklist

- [ ] Initialize with optional seed
- [ ] Select random action from `state.valid_moves`
- [ ] Handle empty valid_moves gracefully (return 0)
- [ ] Implement no-op learning methods
- [ ] Set `is_on_policy = False` and `needs_replay_learning = False`

