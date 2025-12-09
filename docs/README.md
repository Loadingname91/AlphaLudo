# Documentation Index

This directory contains all technical documentation for the Curriculum-Based Ludo RL project.

## Quick Links

- **Getting Started**: See [../README.md](../README.md) - Main project documentation
- **Project Status**: See [../PROJECT_STATUS.md](../PROJECT_STATUS.md) - Current implementation state
- **Visualization**: See [../VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md) - Watch agents play

## Core Documentation

### Implementation Guides

| Document | Description |
|----------|-------------|
| [implementationChecklist.md](implementationChecklist.md) | Complete implementation status with all components |
| [CURRICULUM_IMPLEMENTATION.md](CURRICULUM_IMPLEMENTATION.md) | Curriculum learning approach and training guide |
| [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md) | GPU setup and training acceleration |

### Performance Analysis

| Document | Description |
|----------|-------------|
| [comparison_with_reference_repo.md](comparison_with_reference_repo.md) | Comparison with reference Ludo_Game_AI repository |
| [COMBINED_STATE_ANALYSIS.md](COMBINED_STATE_ANALYSIS.md) | Analysis of combined state space approaches |
| [STATE_ABSTRACTION_GAP_ANALYSIS.md](STATE_ABSTRACTION_GAP_ANALYSIS.md) | Gap analysis between docs and implementation |

### Visualization

| Document | Description |
|----------|-------------|
| [STANDARD_BOARD_VISUALIZER.md](STANDARD_BOARD_VISUALIZER.md) | Classic Ludo board visualizer documentation |

## Technical Documentation

### Agent Implementations

Location: `agents/`

| Document | Agent Type | Status |
|----------|------------|--------|
| [duelingDQNMethodology.md](agents/duelingDQNMethodology.md) | Dueling Double DQN | ✅ Primary agent |
| [dqnAgentMethodology.md](agents/dqnAgentMethodology.md) | Standard DQN | ✅ Baseline |
| [tabularQLearningMethodology.md](agents/tabularQLearningMethodology.md) | Tabular Q-learning | ✅ Classic RL |
| [ruleBasedAgentMethodology.md](agents/ruleBasedAgentMethodology.md) | Heuristic agent | ✅ Rule-based |
| [randomAgentMethodology.md](agents/randomAgentMethodology.md) | Random agent | ✅ Baseline |

### State Abstraction

Location: `stateAbstraction/`

| Document | Abstraction Type | Description |
|----------|------------------|-------------|
| [README.md](stateAbstraction/README.md) | Overview | State abstraction overview |
| [orthogonalState.md](stateAbstraction/orthogonalState.md) | Orthogonal | Orthogonal state representation |
| [augmentedRawState.md](stateAbstraction/augmentedRawState.md) | Augmented Raw | Raw states with augmentation |
| [potentialBasedState.md](stateAbstraction/potentialBasedState.md) | Potential-Based | Potential function states |
| [zoneBasedState.md](stateAbstraction/zoneBasedState.md) | Zone-Based | Zone-based abstraction |
| [combinedState.md](stateAbstraction/combinedState.md) | Combined | Hybrid approach |

### Game Mechanics

Location: `gameLogic/`

| Document | Description |
|----------|-------------|
| [boardPhysics.md](gameLogic/boardPhysics.md) | Board physics and game mechanics |

### Research Methodology

Location: `researchMethodology/`

| Document | Description |
|----------|-------------|
| [experimentalSetup.md](researchMethodology/experimentalSetup.md) | Experimental methodology and setup |

## Archived Documents

These documents are kept for historical reference but are not part of the current implementation:

| Document | Reason |
|----------|--------|
| [ARCHIVED_agent_based_roadmap.md](ARCHIVED_agent_based_roadmap.md) | Original agent-based approach (not used) |
| [ARCHIVED_verificationReport.md](ARCHIVED_verificationReport.md) | Historical verification report |

## Document Organization

### By Topic

**Getting Started**
1. [../README.md](../README.md) - Start here
2. [../PROJECT_STATUS.md](../PROJECT_STATUS.md) - Current state
3. [implementationChecklist.md](implementationChecklist.md) - What's implemented

**Training & Evaluation**
1. [CURRICULUM_IMPLEMENTATION.md](CURRICULUM_IMPLEMENTATION.md) - Training guide
2. [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md) - GPU setup
3. [../VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md) - Watch agents

**Understanding the Approach**
1. [CURRICULUM_IMPLEMENTATION.md](CURRICULUM_IMPLEMENTATION.md) - Curriculum learning
2. State abstraction docs in `stateAbstraction/`
3. Agent methodologies in `agents/`

**Performance & Analysis**
1. [comparison_with_reference_repo.md](comparison_with_reference_repo.md)
2. [COMBINED_STATE_ANALYSIS.md](COMBINED_STATE_ANALYSIS.md)
3. [STATE_ABSTRACTION_GAP_ANALYSIS.md](STATE_ABSTRACTION_GAP_ANALYSIS.md)

### By Role

**For Users/Players**
- [../README.md](../README.md)
- [../VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md)
- [../PROJECT_STATUS.md](../PROJECT_STATUS.md)

**For Researchers**
- [CURRICULUM_IMPLEMENTATION.md](CURRICULUM_IMPLEMENTATION.md)
- [comparison_with_reference_repo.md](comparison_with_reference_repo.md)
- State abstraction docs (`stateAbstraction/`)
- Research methodology (`researchMethodology/`)

**For Developers**
- [implementationChecklist.md](implementationChecklist.md)
- Agent methodologies (`agents/`)
- [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)
- Game logic (`gameLogic/`)

## Contributing to Documentation

When adding new documentation:

1. **Naming Convention**:
   - Use descriptive names: `featureName.md`
   - Use UPPERCASE for major docs: `IMPLEMENTATION_GUIDE.md`
   - Prefix archived docs: `ARCHIVED_oldDoc.md`

2. **Update This Index**:
   - Add link to appropriate section
   - Include brief description
   - Mark status (✅ Complete, 🚧 In Progress, ☐ Planned)

3. **Cross-Reference**:
   - Link to related documents
   - Reference source files where applicable
   - Keep links relative

## Questions?

- **Installation**: See [../README.md](../README.md)
- **Training**: See [CURRICULUM_IMPLEMENTATION.md](CURRICULUM_IMPLEMENTATION.md)
- **Visualization**: See [../VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md)
- **GPU Setup**: See [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)

---

Last Updated: December 2025
