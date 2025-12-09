"""
Gymnasium environment registration for Ludo.

Curriculum-based environments for progressive learning.
"""

from gymnasium.envs.registration import register

# Unified Ludo Environments (used in tests and advanced training)
register(
    id="UnifiedLudo-2Tokens-v0",
    entry_point="rl_agent_ludo.environment.unifiedLudoEnv:UnifiedLudoEnv2Tokens",
    max_episode_steps=10000,
)

register(
    id="UnifiedLudo-4Tokens-v0",
    entry_point="rl_agent_ludo.environment.unifiedLudoEnv:UnifiedLudoEnv4Tokens",
    max_episode_steps=10000,
)

# Curriculum Level Environments (used in main training pipeline)
# Level 1-5 environments are imported directly when needed
# Example: from rl_agent_ludo.environment.level1_simple import Level1SimpleLudo
