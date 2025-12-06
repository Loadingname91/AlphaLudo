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

# Import Unified DQN agent if available
try:
    from .unifiedDQNAgent import (
        UnifiedDQNAgent,
        UnifiedDQNNetwork,
        create_unified_dqn_agent_2tokens,
        create_unified_dqn_agent_4tokens,
    )
    __all__.extend([
        'UnifiedDQNAgent',
        'UnifiedDQNNetwork',
        'create_unified_dqn_agent_2tokens',
        'create_unified_dqn_agent_4tokens',
    ])
except ImportError:
    pass

