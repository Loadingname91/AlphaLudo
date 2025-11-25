"""
Pillar 3: AgentRegistry (Factory Pattern)

Factory for creating agent instances from configuration.
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
        Register a new agent type.
        
        Args:
            name: Agent type name
            agent_class: Agent class (must be subclass of Agent)
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
        
        Returns:
            List of available agent type names
        """
        return list(cls._agent_classes.keys())
