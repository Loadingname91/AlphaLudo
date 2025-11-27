# Extending the Framework: Adding New Agents

This guide explains how to create and register custom agent types without modifying the core framework code.

## Overview

The framework uses a factory pattern (`AgentRegistry`) to create agents from configuration. Built-in agents are automatically registered, but you can add your own agents by:

1. Creating a class that inherits from `Agent`
2. Implementing required methods
3. Registering your agent with `AgentRegistry`
4. Using it in configuration files or code

## Quick Start

Here's a minimal example of a custom agent:

```python
from rl_agent_ludo.agents.base_agent import Agent
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from rl_agent_ludo.utils.state import State

class MyCustomAgent(Agent):
    def __init__(self, my_param=10):
        self.my_param = my_param
    
    def act(self, state: State) -> int:
        # Your action selection logic
        return state.valid_moves[0]
    
    @property
    def is_on_policy(self) -> bool:
        return False  # or True for on-policy agents
    
    @property
    def needs_replay_learning(self) -> bool:
        return False  # or True if using replay buffer

# Register the agent
AgentRegistry.register_agent('my_custom', MyCustomAgent)

# Now use it
config = {'type': 'my_custom', 'my_param': 20}
agent = AgentRegistry.create_agent(config)
```

## Step-by-Step Guide

### Step 1: Create Your Agent Class

Create a new Python file (e.g., `my_agent.py`) and define your agent class:

```python
from rl_agent_ludo.agents.base_agent import Agent
from rl_agent_ludo.utils.state import State
from typing import Optional

class MyAgent(Agent):
    """Your agent description here."""
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """
        Initialize your agent.
        
        All parameters from config (except 'type') are passed as kwargs.
        Accept **kwargs to be compatible with config system.
        """
        self.seed = seed
        # Initialize your agent's internal state here
    
    # Required: Implement these abstract methods
    @property
    def is_on_policy(self) -> bool:
        """
        Return True for on-policy agents (PPO, MCTS),
        False for off-policy agents (Q-Learning, DQN).
        """
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """
        Return True if agent uses experience replay (Q-Learning, DQN),
        False otherwise.
        """
        return False
    
    def act(self, state: State) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current State object with:
                - state.valid_moves: List of valid action indices
                - state.player_pieces: Your piece positions
                - state.enemy_pieces: Opponent piece positions
                - state.dice_roll: Current dice value (1-6)
                - state.abstract_state: Abstract state tuple (for tabular methods)
                - state.full_vector: Feature vector (for neural networks)
        
        Returns:
            Action index (must be in state.valid_moves)
        """
        # Your action selection logic
        # Must return an index from state.valid_moves
        return state.valid_moves[0]
```

### Step 2: Implement Learning Methods (if needed)

If your agent learns, implement the appropriate methods:

**For off-policy agents (Q-Learning, DQN):**
```python
def push_to_replay_buffer(self, state: State, action: int, reward: float,
                         next_state: State, done: bool, **kwargs) -> None:
    """Store experience in replay buffer."""
    # Store (state, action, reward, next_state, done) for later learning
    pass

def learn_from_replay(self, *args, **kwargs) -> None:
    """Learn from stored experiences."""
    # Sample from replay buffer and update your agent
    pass
```

**For on-policy agents (PPO):**
```python
def learn_from_rollout(self, rollout_buffer: List[Dict], *args, **kwargs) -> None:
    """Learn from a complete rollout."""
    # Process the rollout and update policy
    pass
```

**Optional: Episode end callback**
```python
def on_episode_end(self) -> None:
    """Called at the end of each episode."""
    # Useful for epsilon decay, logging, etc.
    pass
```

### Step 3: Register Your Agent

Register your agent before using it. You can do this in several ways:

**Option A: Register in your script**
```python
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from my_agent import MyAgent

# Register before creating agents
AgentRegistry.register_agent('my_agent', MyAgent)

# Now use it
config = {'type': 'my_agent', 'seed': 42}
agent = AgentRegistry.create_agent(config)
```

**Option B: Register at module import**
```python
# In my_agent.py, at the bottom:
from rl_agent_ludo.agents.agent_registry import AgentRegistry
AgentRegistry.register_agent('my_agent', MyAgent)
```

**Option C: Create a registration module**
```python
# In agents/custom_agents.py:
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from .my_agent import MyAgent
from .another_agent import AnotherAgent

def register_custom_agents():
    """Register all custom agents."""
    AgentRegistry.register_agent('my_agent', MyAgent)
    AgentRegistry.register_agent('another_agent', AnotherAgent)

# Call this before using agents
register_custom_agents()
```

### Step 4: Use Your Agent

Once registered, use your agent in configuration files or code:

**In YAML config:**
```yaml
agent:
  type: "my_agent"
  seed: 42
  my_param: 20
```

**In Python code:**
```python
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from my_agent import MyAgent

# Register first
AgentRegistry.register_agent('my_agent', MyAgent)

# Then create
config = {'type': 'my_agent', 'seed': 42, 'my_param': 20}
agent = AgentRegistry.create_agent(config)
```

## Required vs Optional Methods

### Required Methods (Abstract)

These **must** be implemented:

- `act(state: State) -> int`: Action selection
- `is_on_policy` (property)`: Whether agent is on-policy
- `needs_replay_learning` (property)`: Whether agent uses replay buffer

### Optional Methods (Have Default Implementations)

These can be overridden if needed:

- `learn_from_replay()`: For off-policy agents using replay
- `learn_from_rollout(rollout_buffer)`: For on-policy agents
- `push_to_replay_buffer(...)`: For agents with replay buffers
- `on_episode_end()`: Called at episode end (useful for epsilon decay)
- `save(filepath)`: Save agent state to file
- `load(filepath)`: Load agent state from file
- `supports_score_debug` (property): Enable score debugging
- `get_last_score_debug()`: Return debug information

## Common Patterns

### Pattern 1: Simple Non-Learning Agent

```python
class SimpleAgent(Agent):
    def __init__(self, strategy='first'):
        self.strategy = strategy
    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        return False
    
    def act(self, state: State) -> int:
        if self.strategy == 'first':
            return state.valid_moves[0]
        elif self.strategy == 'random':
            import random
            return random.choice(state.valid_moves)
        else:
            return state.valid_moves[0]
```

### Pattern 2: Off-Policy Learning Agent (Q-Learning style)

```python
from collections import defaultdict
import numpy as np

class MyQLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, epsilon=0.1, **kwargs):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.last_state = None
        self.last_action = None
    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        return True
    
    def act(self, state: State) -> int:
        import random
        if random.random() < self.epsilon:
            return random.choice(state.valid_moves)
        else:
            # Greedy action
            state_key = state.abstract_state
            q_values = self.q_table[state_key]
            best_action = max(state.valid_moves, key=lambda a: q_values[a])
            self.last_state = state_key
            self.last_action = best_action
            return best_action
    
    def push_to_replay_buffer(self, state, action, reward, next_state, done, **kwargs):
        # Q-learning update
        if self.last_state is not None:
            next_q = max(self.q_table[next_state.abstract_state])
            current_q = self.q_table[self.last_state][self.last_action]
            new_q = current_q + self.learning_rate * (reward + 0.9 * next_q - current_q)
            self.q_table[self.last_state][self.last_action] = new_q
    
    def learn_from_replay(self):
        # Called after push_to_replay_buffer
        pass
    
    def on_episode_end(self):
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.999)
```

### Pattern 3: On-Policy Agent (PPO style)

```python
class MyPPOAgent(Agent):
    def __init__(self, **kwargs):
        self.policy_net = None  # Your policy network
        self.rollout_buffer = []
    
    @property
    def is_on_policy(self) -> bool:
        return True
    
    @property
    def needs_replay_learning(self) -> bool:
        return False
    
    def act(self, state: State) -> int:
        # Sample action from policy
        action_probs = self.policy_net(state.full_vector)
        action = np.random.choice(len(action_probs), p=action_probs)
        return state.valid_moves[action]
    
    def learn_from_rollout(self, rollout_buffer, *args, **kwargs):
        # Update policy from rollout
        # Process rollout_buffer and update policy_net
        pass
```

### Pattern 4: Agent with Score Debugging

```python
class DebuggableAgent(Agent):
    def __init__(self, debug_scores=False, **kwargs):
        self.debug_scores = debug_scores
        self._last_score_debug = None
    
    @property
    def supports_score_debug(self) -> bool:
        return self.debug_scores
    
    def get_last_score_debug(self):
        return self._last_score_debug
    
    def act(self, state: State) -> int:
        # Calculate scores for each action
        scores = {}
        for action in state.valid_moves:
            scores[action] = self._calculate_score(state, action)
        
        best_action = max(scores, key=scores.get)
        
        if self.debug_scores:
            self._last_score_debug = {
                'action': best_action,
                'scores': scores,
                'total_score': scores[best_action]
            }
        
        return best_action
```

## Best Practices

1. **Accept `**kwargs` in `__init__`**: This makes your agent compatible with the config system, even if you don't use all parameters.

2. **Use type hints**: Helps with IDE support and documentation.

3. **Document your agent**: Add docstrings explaining what your agent does and its hyperparameters.

4. **Handle edge cases**: Always check if `state.valid_moves` is empty and handle gracefully.

5. **Seed management**: If your agent uses randomness, accept a `seed` parameter and set it.

6. **State representation**: Decide whether to use `state.abstract_state` (for tabular methods) or `state.full_vector` (for neural networks).

7. **Error handling**: Validate inputs and provide clear error messages.

## Testing Your Agent

Create a simple test script:

```python
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from rl_agent_ludo.environment.ludo_env import LudoEnv
from my_agent import MyAgent

# Register agent
AgentRegistry.register_agent('my_agent', MyAgent)

# Create environment
env = LudoEnv()

# Create agent
config = {'type': 'my_agent', 'seed': 42}
agent = AgentRegistry.create_agent(config)

# Test action selection
state = env.reset()
action = agent.act(state)
print(f"Selected action: {action}")

# Test training loop
for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        if agent.needs_replay_learning:
            agent.push_to_replay_buffer(state, action, reward, state, done)
            agent.learn_from_replay()
    if hasattr(agent, 'on_episode_end'):
        agent.on_episode_end()
```

## Troubleshooting

**Problem**: `ValueError: Unknown agent type`
- **Solution**: Make sure you registered your agent before calling `create_agent()`

**Problem**: `TypeError: agent_class must be subclass of Agent`
- **Solution**: Ensure your class inherits from `Agent`

**Problem**: Agent not learning
- **Solution**: Check that `needs_replay_learning` or `is_on_policy` is set correctly, and implement the appropriate learning methods

**Problem**: Config parameters not passed to agent
- **Solution**: Make sure your `__init__` accepts `**kwargs` or explicitly lists all parameters

## Examples in the Codebase

Look at existing agents for reference:

- **Simple agent**: `src/rl_agent_ludo/agents/random_agent.py`
- **Off-policy learning**: `src/rl_agent_ludo/agents/QLearning_agent.py`
- **Deep learning**: `src/rl_agent_ludo/agents/dqn_agent.py`
- **Heuristic**: `src/rl_agent_ludo/agents/rule_based_heuristic_agent.py`

## See Also

- `src/rl_agent_ludo/agents/base_agent.py` - Base class definition
- `src/rl_agent_ludo/agents/agent_registry.py` - Registration system
- `src/rl_agent_ludo/agents/agent_template.py` - Template file with examples

