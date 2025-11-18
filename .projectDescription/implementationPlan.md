```table-of-contents
```

**Date:** 2025-11-13
**Tags:** #UoUAI

---

### **Project Blueprint: A Hierarchical Analysis of Reinforcement Learning for Ludo**

#### 1.0 Executive Summary

This document outlines a formal blueprint for a modular, research-grade project to implement, test, and comparatively analyze a hierarchy of reinforcement learning (RL) agents for the game of Ludo.

The project's primary goal is not to simply create a winning agent, but to build a robust experimental framework that allows for a deep, nuanced analysis of the trade-offs, biases, and "gotchas" associated with each major RL algorithm. The architecture is designed for modularity, reproducibility, and the explicit separation of concerns, ensuring that the final analysis is grounded in rigorous, empirical data.

The implementation will proceed in phases, from a simple `RandomAgent` baseline to advanced search-based agents, mirroring the "state-of-the-art" as identified in the accompanying research survey.

---

#### 2.0 Core System Architecture

The system is designed around six key pillars, which form a clean experimental backbone.

- **Pillar 1: `LudoEnv` (Environment Abstraction Layer)**
    
    - **Responsibility:** To act as the sole "hardware abstraction layer" (HAL) between the project and the `ludopy` library. It provides a clean, Gym-like `step()`/`reset()` interface, abstracting away the specifics of the `ludopy` API 2.
        
    - **Key Attributes:** `game` (the raw `ludopy` instance), `opponent_agents` list, `opponent_schedule` (for curriculum learning), `player_id_map` (to handle dynamic player IDs).
        
    - **Key Methods:** `reset()`, `step()`, `get_valid_actions()`, `_get_observation()`, `_execute_action()`, `_get_full_state_vector()` (State Abstraction), `_get_abstract_state()` (State Abstraction).
        
- **Pillar 1.5: `State` (Data Transfer Object)**
    
    - **Responsibility:** An immutable data structure (dataclass) that passes all necessary state information from the `LudoEnv` to the `Agent`. This simplifies the API and ensures data consistency.
        
    - **Key Attributes:** `full_vector` (NumPy array for NNs), `abstract_state` (hashable tuple for tables), `valid_moves` (list), `dice_roll` (int).
        
- **Pillar 2: `RewardShaper` (Strategy Pattern)**
    
    - **Responsibility:** Implements the Strategy design pattern for reward logic. It completely decouples the environment's `step` function from the reward heuristic being tested (e.g., 'sparse', 'dense' 3, or 'decoupled-ila' 4).
        
    - **Key Attributes:** `schema` (string identifier).
        
    - **Key Methods:** `get_reward()` (processes game events and returns a net reward and a dictionary of ILA components).
        
- **Pillar 3: `Agent` (Interface) & `AgentRegistry` (Factory)**
    
    - **Responsibility:** The `Agent` class is an abstract base class (interface) that defines the methods all agents must implement. The `AgentRegistry` is a Factory pattern that constructs agent objects from a configuration file.
        
    - **`Agent` Key Attributes:** `is_on_policy` (boolean), `needs_replay_learning` (boolean).
        
    - **`Agent` Key Methods:** `act()`, `learn_from_replay()`, `learn_from_rollout()`, `push_to_replay_buffer()`.
        
- **Pillar 4: Decoupled Metrics & Analysis**
    
    - **Responsibility:** To separate data _collection_ from data _analysis_, ensuring the training loop remains lightweight.
        
    - **Component: `MetricsTracker`:** A simple class with no analysis dependencies (e.g., pandas, matplotlib). It collects raw data (lists, dicts) and saves to disk (JSON/CSV).
        
    - **Component: `analysis.py`:** A separate, offline script that loads the raw data, performs `pandas` transformations, and generates all 5-point analysis plots using `matplotlib`/`seaborn`.
        
- **Pillar 5: `Trainer` (Orchestrator)**
    
    - **Responsibility:** The main application orchestrator. It manages the training loop, environment-agent interaction, and logging.
        
    - **Key Attributes:** `env`, `agent`, `config`, `logger`, `experiment_logger` (e.g., TensorBoard/WandB).
        
    - **Key Methods:** `run()` (selects loop type), `_set_seeds()` (for reproducibility), `_run_on_policy_loop()` (for PPO), `_run_off_policy_loop()` (for Q-Learning/DQN).
        
- **Pillar 6: `pytest` Validation Harness**
    
    - **Responsibility:** To ensure code correctness, API stability, and reproducibility of core components.
        
    - **Key Tests:** `test_env_api` (asserts `step`/`reset` return types), `test_state_abstraction` (asserts vector/tuple formats), `test_reward_shaping` (asserts correct rewards for mock events).
        

---

#### 3.0 Hierarchical Implementation Plan

The project will be implemented in phases, with each new agent representing a step up in complexity and theoretical approach. Each phase concludes with a full execution of the 5-point analysis framework.

- **Phase 0: Baseline**
    
    - **Agent:** `RandomAgent`
        
    - **Purpose:** Validate the `Trainer`, `LudoEnv`, and `MetricsTracker`. Establishes the ~25% win-rate baseline.

- **Phase 0.5: Rule Based Heuristic Agent**
    
    - **Agent:** `RuleBasedHeuristicAgent`
        
    - **Purpose:** To validate the `Trainer`, `LudoEnv`, and `MetricsTracker`. SHould ideally be able to beat the `RandomAgent` by a significant margin.

    - **Key Challenge**: Comming with a heuristic that is not too complex, but still able to beat the `RandomAgent` by a significant margin.

- **Phase 1: Tabular Q-Learning**
    
    - **Agent:** `TabularQAgent`
        
    - **Purpose:** To solve the problem with manual state abstraction, as seen in `NDurocher/YARAL` 555555555. This confronts the state-space explosion problem directly.
        
    - **Key Challenge:** Engineering the `_get_abstract_state()` function.
        
- **Phase 1.5: Advanced Tabular**
    
    - **Agent:** `TDAgent`
        
    - **Purpose:** To implement TD($\lambda$) 6 and eligibility traces. This tests the impact of improved temporal credit assignment on a sparse-reward, tabular agent7.
        
- **Phase 2: Deep Q-Network**
    
    - **Agent:** `DQNAgent`
        
    - **Purpose:** To use neural networks for function approximation, eliminating manual state abstraction8. This phase, inspired by `MehranSangrasi/AI-Ludo` 9, focuses heavily on analyzing the impact of **dense reward shaping** 10and stabilization techniques (Replay, Target Nets)11.
        
- **Phase 3: Policy Gradient**
    
    - **Agent:** `PPOAgent`
        
    - **Purpose:** To implement a state-of-the-art, on-policy algorithm. Proximal Policy Optimization (PPO) is known for its stability in high-variance environments 12, making it a prime candidate for Ludo's stochastic (dice) nature.
        
- **Phase 4: Search-Based (State-of-the-Art)**
    
    - **Agent:** `MCTSAgent`
        
    - **Purpose:** To implement an "AlphaLudo" style agent based on `rajtilakls2510/solving_ludo1`13. This combines Monte-Carlo Tree Search (MCTS) 14with a neural network, an approach that natively handles stochasticity and learns from sparse win/loss signals15.
        

---

#### 4.0 In-Depth Analysis Framework

Upon completion of each phase, a standardized 5-point analysis will be performed to generate the data for the final report.

1. **Policy & Behavioral Analysis:** What _kind_ of player was created? (e.g., Aggression, Defense, Efficiency metrics).
    
2. **Stability & Convergence Analysis:** How stable was the learning? (e.g., Q-value variance, win-rate confidence intervals).
    
3. **Robustness & Generalization Analysis:** How well does the agent adapt? (e.g., The "Opponent Swap Test" to expose IQL flaws).
    
4. **Computational & Scalability Analysis:** What is the engineering cost? (e.g., Sample Efficiency, Inference Time per Move).
    
5. **Hyperparameter Sensitivity Analysis:** How "fiddly" is the algorithm? (e.g., Plotting win rates vs. `alpha` or `clip_range`).
    

---

#### 5.0 Projected Analysis Summary

The final deliverable will be a comprehensive report synthesized from the `analysis.py` outputs. The central artifact will be a comparative summary table, projected to look as follows:

|**Algorithm**|**Win Rate (vs. Random)**|**Convergence Speed**|**Inference Time (ms)**|**Stability (Win Rate CI)**|**Policy Profile**|
|---|---|---|---|---|---|
|`RandomAgent`|~25%|N/A|< 1ms|±0.0%|Random|
|`TabularQ (Sparse)`|45-55%|Medium (~15k ep)|< 1ms|±8.5%|Defensive|
|`TD(λ) (Sparse)`|50-60%|Medium-Fast (~10k ep)|< 1ms|±6.2%|Defensive|
|`DQN (Sparse)`|40-50% (Unstable)|Slow / Unreliable|< 1ms|±20.1%|Erratic|
|`DQN (Dense)`|80-85%|Very Fast (~2k ep)|< 1ms|±4.1%|**Aggressive**|
|`PPO (Dense)`|78-83%|Fast (~3k ep)|< 1ms|**±2.3%**|Aggressive|
|`MCTS (Sparse)`|**90-95%**|Slow (~50k ep)|**> 500ms**|**±1.9%**|**Robust**|

