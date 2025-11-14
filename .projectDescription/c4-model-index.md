# C4 Model Documentation - Master Index

## Overview

This directory contains a complete C4 model documentation of the **RL Agent Ludo** system, structured in four levels from high-level system context to detailed code interactions.

The C4 model uses four hierarchical levels:

1. **Level 1: System Context** - Shows the system and its users
2. **Level 2: Container** - Shows high-level technical building blocks
3. **Level 3: Component** - Shows components within containers
4. **Level 4: Code** - Shows classes, methods, and code-level interactions

---

## Documentation Structure

### 📍 Level 1: System Context
**File**: [`c4-level1-context.md`](./c4-level1-context.md)

**Purpose**: Highest level view showing actors, external systems, and system boundaries.

**Key Elements**:
- Human actors (RL Researcher, Data Scientist, Developer)
- External systems (Ludopy Library, ML Frameworks, Logging Tools)
- System boundary definition
- Key interactions between actors and systems

**When to Use**: Understanding the overall system purpose and who interacts with it.

**Next Level**: [Level 2: Container Diagram](./c4-level2-container.md)

---

### 📦 Level 2: Container Diagram
**File**: [`c4-level2-container.md`](./c4-level2-container.md)

**Purpose**: Shows high-level technical building blocks (applications, data stores, file systems).

**Key Containers**:
1. **Training Application** - Main orchestrator for training workflows
2. **Agent Library** - Contains all RL agent implementations
3. **Environment Library** - HAL abstraction for Ludo game
4. **Metrics Tracker** - Collects and stores raw training metrics
5. **Analysis Service** - Offline analysis and reporting
6. **Metrics Storage** - JSON/CSV file storage
7. **Model Store** - File system storage for trained models

**Technology Decisions**:
- Python as primary language
- Separation of metrics collection and analysis
- File-based storage (JSON/CSV)
- External ML frameworks (PyTorch/TensorFlow)
- External logging (TensorBoard/WandB)

**When to Use**: Understanding the high-level architecture and technology stack.

**Previous Level**: [Level 1: System Context](./c4-level1-context.md)  
**Next Level**: [Level 3: Component Diagrams](./c4-level3-components.md)

---

### 🔧 Level 3: Component Diagrams
**File**: [`c4-level3-components.md`](./c4-level3-components.md)

**Purpose**: Shows the internal structure of containers, including components and their interactions.

**Component Diagrams Included**:

1. **Training Application Container**
   - Trainer (main orchestrator)
   - ConfigManager (configuration loading)
   - OnPolicyLoop (PPO training)
   - OffPolicyLoop (Q-Learning/DQN training)
   - SeedManager (reproducibility)
   - Logger (console logging)

2. **Agent Library Container**
   - Agent (abstract interface)
   - AgentRegistry (factory pattern)
   - Concrete agents: RandomAgent, TabularQAgent, TDAgent, DQNAgent, PPOAgent, MCTSAgent
   - Supporting components: ReplayBuffer, TargetNetwork, PolicyNetwork, ValueNetwork, MCTSTree

3. **Environment Library Container**
   - LudoEnv (Gym-like interface)
   - State (immutable DTO)
   - StateAbstractor (state conversion)
   - OpponentManager (opponent handling)
   - ValidActionsManager (action filtering)
   - RewardShaper (strategy pattern)
   - Reward strategies: SparseReward, DenseReward, ILAReward

4. **Metrics & Analysis Containers**
   - **Metrics Tracker**: MetricsTracker, EpisodeRecorder, StepRecorder, MetricExporter
   - **Analysis Service**: AnalysisRunner, DataLoader, 5 Analyzers (Policy, Stability, Robustness, Computational, Hyperparameter), PlotGenerator, ReportGenerator

**When to Use**: Understanding the internal structure of containers and component responsibilities.

**Previous Level**: [Level 2: Container Diagram](./c4-level2-container.md)  
**Next Level**: [Level 4: Code Diagrams](./c4-level4-code.md)

---

### 💻 Level 4: Code Diagrams
**File**: [`c4-level4-code.md`](./c4-level4-code.md)

**Purpose**: Shows detailed code-level interactions, sequence diagrams, and class structures.

**Scenarios Covered**:

1. **On-Policy Training Loop (PPO Agent)**
   - Complete sequence diagram for PPO training
   - Code-level class structure for PPOAgent
   - Rollout collection and learning phases

2. **Off-Policy Training Loop (DQN Agent)**
   - Sequence diagram for DQN with experience replay
   - Code-level class structure for DQNAgent
   - Replay buffer and target network updates

3. **State Abstraction Process**
   - Sequence diagram for state creation
   - Code-level StateAbstractor class
   - Full vector vs. abstract state creation

4. **Reward Shaping Process**
   - Sequence diagram for reward calculation
   - Code-level RewardShaper strategy pattern
   - Sparse, Dense, and ILA reward implementations

5. **Metrics Collection Process**
   - Sequence diagram for metrics logging and export
   - Code-level MetricsTracker class
   - Episode and step recording

6. **Analysis Execution Process**
   - Sequence diagram for 5-point analysis framework
   - Code-level AnalysisRunner and analyzer classes
   - Report generation flow

**Code-Level Details**:
- Class structures with methods and attributes
- Function signatures
- Implementation details
- Agent interface hierarchy

**When to Use**: Understanding exact code execution flows and implementation details.

**Previous Level**: [Level 3: Component Diagrams](./c4-level3-components.md)

---

## Navigation Guide

### Top-Down Approach (Recommended for New Readers)
1. Start with [Level 1: System Context](./c4-level1-context.md) to understand the overall system
2. Move to [Level 2: Container Diagram](./c4-level2-container.md) to see high-level architecture
3. Dive into [Level 3: Component Diagrams](./c4-level3-components.md) to understand container internals
4. Finish with [Level 4: Code Diagrams](./c4-level4-code.md) for implementation details

### Bottom-Up Approach (For Developers)
1. Start with [Level 4: Code Diagrams](./c4-level4-code.md) to understand implementation
2. Move to [Level 3: Component Diagrams](./c4-level3-components.md) to see component relationships
3. Review [Level 2: Container Diagram](./c4-level2-container.md) to understand container boundaries
4. Finish with [Level 1: System Context](./c4-level1-context.md) to see the bigger picture

### Quick Reference
- **System Overview**: [Level 1](./c4-level1-context.md)
- **Architecture**: [Level 2](./c4-level2-container.md)
- **Component Design**: [Level 3](./c4-level3-components.md)
- **Implementation Details**: [Level 4](./c4-level4-code.md)

---

## Mapping to Implementation Plan

The C4 model documentation aligns with the implementation plan's six pillars:

| C4 Level | Implementation Plan Pillar | Container/Component |
|----------|---------------------------|-------------------|
| Level 2 | Pillar 1: LudoEnv | Environment Library Container |
| Level 3 | Pillar 1.5: State | State Component (in Environment Library) |
| Level 3 | Pillar 2: RewardShaper | RewardShaper Component (in Environment Library) |
| Level 3 | Pillar 3: Agent & Registry | Agent Library Container |
| Level 3 | Pillar 4: Metrics & Analysis | Metrics Tracker + Analysis Service Containers |
| Level 2 | Pillar 5: Trainer | Training Application Container |
| Level 3 | Pillar 6: pytest | (Not shown in C4, testing infrastructure) |

---

## Key Design Patterns Used

1. **Strategy Pattern**: RewardShaper (Sparse, Dense, ILA strategies)
2. **Factory Pattern**: AgentRegistry (creates agent instances)
3. **Abstract Factory**: Agent interface with concrete implementations
4. **DTO Pattern**: State (immutable data transfer object)
5. **HAL Pattern**: LudoEnv (hardware abstraction layer)

---

## Technology Stack Summary

- **Language**: Python
- **ML Frameworks**: PyTorch/TensorFlow (for DQN, PPO, MCTS)
- **Game Engine**: Ludopy (external library)
- **Data Storage**: JSON/CSV files
- **Analysis**: Pandas, Matplotlib, Seaborn
- **Logging**: TensorBoard/WandB
- **Testing**: pytest

---

## Related Documentation

- [Implementation Plan](../implementationPlan.md) - Detailed project blueprint
- [Research Documentation](./research/) - Research findings and references

---

## Diagram Format

All diagrams use **Mermaid** syntax, which is supported by:
- GitHub (native rendering)
- GitLab (native rendering)
- Many Markdown viewers (via plugins)
- VS Code (with Mermaid extension)

To view diagrams:
1. Use a Markdown viewer with Mermaid support
2. Copy diagram code to [Mermaid Live Editor](https://mermaid.live/)
3. Export as PNG/SVG for presentations

---

## Maintenance

When updating the architecture:

1. **Update Level 1** if actors or external systems change
2. **Update Level 2** if containers are added/removed or technology changes
3. **Update Level 3** if components change within containers
4. **Update Level 4** if code structures or execution flows change

Always maintain consistency across all four levels.

---

**Last Updated**: 2025-11-13  
**Version**: 1.0  
**Status**: Complete C4 Model Documentation

