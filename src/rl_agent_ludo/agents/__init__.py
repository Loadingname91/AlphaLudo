"""
RL Agents for Ludo.

This package contains the agents used in the curriculum-based training approach.
"""

from .baseAgent import Agent
from .baseline_agents import RandomAgent, GreedyAgent
from .simple_dqn import SimpleDQNAgent

__all__ = [
    'Agent',
    'RandomAgent',
    'GreedyAgent',
    'SimpleDQNAgent',
]
