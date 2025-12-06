# Implementation Roadmap

Step-by-step guide for implementing all 5 agents in order.

## Overview

This roadmap provides a structured approach to implementing each agent, building complexity incrementally. Each agent validates the environment and builds on previous knowledge.

## Prerequisites

- [ ] Gymnasium environment (`LudoEnv`) implemented
- [ ] State abstraction utilities available
- [ ] Reward shaping utilities available
- [ ] Basic training loop structure in place

---

## Phase 1: Random Agent (Baseline)

**Purpose**: Validate environment, establish baseline win rate (~25%)

**Implementation Steps**:
1. Create `agents/randomAgent.py`
2. Inherit from `Agent` base class
3. Implement `act()`: Random choice from `state.valid_moves`
4. Implement no-op learning methods
5. Set `is_on_policy = False`, `needs_replay_learning = False`
6. Handle edge case: Empty `valid_moves` → return 0

**Testing**:
- [ ] Agent selects valid actions
- [ ] Win rate ~25% over 1000 episodes
- [ ] No errors in training loop

**Documentation**: See `docs/agents/randomAgentMethodology.md`

**Estimated Time**: 1-2 hours

---

## Phase 2: Rule-Based Heuristic Agent

**Purpose**: Hand-crafted rules with phase-aware multipliers

**Implementation Steps**:
1. Create `agents/ruleBasedAgent.py`
2. Implement priority rule hierarchy (Win > Capture/Flee > Home Progress > Blockade > Progress)
3. Implement game phase detection (Opening, Midgame, Closing, Critical)
4. Implement risk calculation (`_calculate_probablistic_risk`)
5. Implement contextual multipliers (`_get_contextual_multipler`)
6. Implement move scoring (`_calculate_move_score`)
7. Implement action selection with tie-breaking
8. Handle all edge cases (see methodology)

**Testing**:
- [ ] Agent selects reasonable moves
- [ ] Win rate > 25% (should outperform random)
- [ ] Phase detection works correctly
- [ ] Risk calculation handles safe positions

**Documentation**: See `docs/agents/ruleBasedAgentMethodology.md`

**Dependencies**: `board_analyser.py` utilities (or reimplement from docs)

**Estimated Time**: 4-6 hours

---

## Phase 3: Tabular Q-Learning Agent

**Purpose**: Context-aware Q-learning with state abstraction

**Implementation Steps**:
1. Create `agents/tabularQLearningAgent.py`
2. Implement state abstractor (`LudoStateAbstractor` from docs)
3. Initialize Q-table: `defaultdict(lambda: np.zeros(4))`
4. Implement epsilon-greedy action selection
5. Implement Bellman update with context-aware reward scaling
6. Implement reward scaling multipliers (trailing, leading, neutral)
7. Implement epsilon decay per episode
8. Implement save/load for Q-table
9. Handle all edge cases (see methodology)

**Testing**:
- [ ] Q-table updates correctly
- [ ] Context calculation works
- [ ] Reward scaling applied correctly
- [ ] Win rate improves over training
- [ ] Checkpointing works

**Documentation**: 
- See `docs/agents/tabularQLearningMethodology.md`
- See `docs/stateAbstraction/contextAwareState.md`

**Dependencies**: State abstractor, board analyser utilities

**Estimated Time**: 6-8 hours

---

## Phase 4: DQN Agent

**Purpose**: Deep Q-Network with experience replay

**Implementation Steps**:
1. Create `agents/dqnAgent.py`
2. Implement state abstractor (orthogonal or augmented_raw)
3. Implement neural network (standard DQN architecture)
4. Initialize target network (copy of online network)
5. Implement experience replay buffer (uniform sampling)
6. Implement epsilon-greedy action selection
7. Implement Bellman update with target network
8. Implement target network updates (every N steps)
9. Implement gradient clipping
10. Implement epsilon decay per episode
11. Implement save/load for model checkpoints
12. Handle all edge cases (see methodology)

**Testing**:
- [ ] Network architecture correct
- [ ] Experience replay works
- [ ] Target network updates correctly
- [ ] Win rate improves over training
- [ ] Checkpointing works

**Documentation**: 
- See `docs/agents/dqnAgentMethodology.md`
- See `docs/stateAbstraction/orthogonalState.md` or `augmentedRawState.md`

**Dependencies**: PyTorch, state abstractor, board analyser utilities

**Estimated Time**: 8-10 hours

---

## Phase 5: Dueling DQN Agent

**Purpose**: Dueling architecture + Double Q-Learning + PER

**Implementation Steps**:
1. Create `agents/duelingDQNAgent.py` (or extend DQN)
2. Implement dueling architecture (value + advantage streams)
3. Implement advantage mean subtraction for identifiability
4. Implement double Q-learning (online selects, target evaluates)
5. Implement prioritized experience replay (SumTree)
6. Implement importance sampling weights
7. Implement beta annealing schedule
8. Implement priority updates based on TD-errors
9. Handle all edge cases (see methodology)

**Testing**:
- [ ] Dueling architecture works
- [ ] Double Q-learning reduces overestimation
- [ ] PER samples high-priority transitions
- [ ] Win rate improves over training
- [ ] Checkpointing works

**Documentation**: 
- See `docs/agents/duelingDQNMethodology.md`
- See `docs/agents/dqnAgentMethodology.md` for shared components

**Dependencies**: All DQN dependencies + PER implementation

**Estimated Time**: 10-12 hours

---

## General Implementation Guidelines

### Code Structure

- Follow methodology documentation exactly
- Implement all edge cases documented
- Use type hints
- Add docstrings for all methods
- Handle errors gracefully

### Testing Strategy

1. **Unit Tests**: Test individual methods
2. **Integration Tests**: Test agent-environment interaction
3. **Performance Tests**: Verify win rate improvements
4. **Edge Case Tests**: Test all documented edge cases

### Checkpointing

- Implement `save()` and `load()` methods
- Save all necessary state (weights, hyperparameters, training progress)
- Handle version mismatches gracefully

### Hyperparameters

- Start with defaults from methodology docs
- Tune based on performance
- Document any changes from defaults

### Debugging

- Use `debug_scores` flag for detailed logging
- Log Q-values, action selections, rewards
- Track training metrics (loss, win rate, epsilon)

---

## Success Criteria

All agents are complete when:
- [ ] Implementation matches methodology documentation
- [ ] All edge cases handled
- [ ] Tests pass
- [ ] Win rate meets or exceeds expected performance
- [ ] Checkpointing works
- [ ] Code is clean and well-documented

---

## Next Steps After Implementation

1. Run comprehensive experiments
2. Compare agent performance
3. Analyze state abstractions
4. Tune hyperparameters
5. Document results

