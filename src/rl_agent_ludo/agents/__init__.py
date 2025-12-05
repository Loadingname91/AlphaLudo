"""
RL Agents for Ludo.

Implement agents following the methodologies in docs/agents/
"""

from .dqnAgent import DQNAgent
from .tabularQAgent import TabularQAgent
from .ruleBasedAgent import RuleBasedAgent
from .baseAgent import Agent

__all__ = ['Agent', 'TabularQAgent', 'DQNAgent', 'RuleBasedAgent']

