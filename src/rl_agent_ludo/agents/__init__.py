"""
RL Agents for Ludo.

Implement agents following the methodologies in docs/agents/
"""


from .tabularQAgent import TabularQAgent
from .ruleBasedAgent import RuleBasedAgent
from .baseAgent import Agent

# Import DQN agent if available
try:
    from .dqnAgent import DQNAgent
    __all__ = ['Agent', 'TabularQAgent', 'RuleBasedAgent', 'DQNAgent']
except ImportError:
    __all__ = ['Agent', 'TabularQAgent', 'RuleBasedAgent']

