"""
Agent factory.

Creates agent instances from configuration dictionaries.

Built-in agents are automatically registered. Custom agents can be registered
using the register_agent() method without modifying this file.

Example - Using built-in agents:
    config = {'type': 'random', 'seed': 42}
    agent = AgentRegistry.create_agent(config)

Example - Registering a custom agent:
    from rl_agent_ludo.agents.base_agent import Agent
    from rl_agent_ludo.agents.agent_registry import AgentRegistry
    
    class MyCustomAgent(Agent):
        def act(self, state):
            # Your implementation
            pass
        
        @property
        def is_on_policy(self):
            return False
        
        @property
        def needs_replay_learning(self):
            return True
    
    # Register before creating agents
    AgentRegistry.register_agent('my_custom', MyCustomAgent)
    
    # Now you can use it in config files
    config = {'type': 'my_custom', 'param1': 'value1'}
    agent = AgentRegistry.create_agent(config)

See docs/EXTENDING_AGENTS.md for a complete guide on creating custom agents.
"""

from typing import Dict, Any, Optional

from rl_agent_ludo.agents.rule_based_heuristic_agent import RuleBasedHeuristicAgent
from .base_agent import Agent
from .random_agent import RandomAgent
from .QLearning_agent import QLearningAgent
from .dqn_agent import DQNAgent

class AgentRegistry:
    """
    Factory for creating agent instances from configuration.
    
    Implements Factory pattern to decouple agent creation from configuration.
    
    Built-in agents (random, q_learning, dqn, rule_based_heuristic) are
    automatically available. Custom agents can be registered using
    register_agent() without modifying this file.
    """
    
    # Registry of available agent types
    _agent_classes: Dict[str, type] = {
        'random': RandomAgent,
        'rule_based_heuristic': RuleBasedHeuristicAgent,
        'q_learning': QLearningAgent,
        'dqn': DQNAgent,
        # Rule based  agent : to be implemented in Phase 1
        # 'tabular_q': TabularQAgent,  # To be implemented in Phase 1
        # 'td': TDAgent,  # To be implemented in Phase 1.5
        # 'ppo': PPOAgent,  # To be implemented in Phase 3
        # 'mcts': MCTSAgent,  # To be implemented in Phase 4
    }
    
    @classmethod
    def register_agent(cls, name: str, agent_class: type) -> None:
        """
        Register a new agent type for use in configuration files.
        
        This method allows you to add custom agents without modifying
        agent_registry.py. Register your agent before calling create_agent().
        
        Args:
            name: Agent type name (used in config files as 'type: name')
            agent_class: Agent class (must be subclass of Agent)
        
        Raises:
            TypeError: If agent_class is not a subclass of Agent
        
        Example:
            # Define your custom agent
            class MyAgent(Agent):
                def __init__(self, my_param=10):
                    self.my_param = my_param
                
                def act(self, state):
                    return state.valid_moves[0]
                
                @property
                def is_on_policy(self):
                    return False
                
                @property
                def needs_replay_learning(self):
                    return False
            
            # Register it
            AgentRegistry.register_agent('my_agent', MyAgent)
            
            # Now use it in config or code
            config = {'type': 'my_agent', 'my_param': 20}
            agent = AgentRegistry.create_agent(config)
        
        Note:
            Registration persists for the lifetime of the Python process.
            Register agents before creating instances, typically at module
            import time or in your main script before loading configs.
        """
        if not issubclass(agent_class, Agent):
            raise TypeError(f"agent_class must be subclass of Agent, got {agent_class}")
        cls._agent_classes[name] = agent_class
    
    @classmethod
    def create_agent(cls, config: Dict[str, Any]) -> Agent:
        """
        Create agent instance from configuration.
        
        Args:
            config: Configuration dictionary with at least 'type' key.
                   Additional keys are passed as kwargs to agent constructor.
        
        Returns:
            Agent instance
        
        Raises:
            ValueError: If agent type is not recognized
            TypeError: If agent_type is not registered
        
        Example:
            config = {
                'type': 'random',
                'seed': 42
            }
            agent = AgentRegistry.create_agent(config)
        """
        agent_type = config.get('type', 'random').lower()
        
        if agent_type not in cls._agent_classes:
            available = list(cls._agent_classes.keys())
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {available}"
            )
        
        agent_class = cls._agent_classes[agent_type]
        
        # Extract agent-specific config (everything except 'type')
        agent_config = {k: v for k, v in config.items() if k != 'type'}
        
        # Create agent instance
        try:
            agent = agent_class(**agent_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create agent of type '{agent_type}': {e}"
            ) from e
        
        return agent
    
    @classmethod
    def get_available_agents(cls) -> list:
        """
        Get list of available agent type names.
        
        Includes both built-in agents and any custom agents registered
        via register_agent().
        
        Returns:
            List of available agent type names (e.g., ['random', 'q_learning', 'dqn', ...])
        
        Example:
            available = AgentRegistry.get_available_agents()
            print(f"Available agents: {available}")
        """
        return list(cls._agent_classes.keys())
