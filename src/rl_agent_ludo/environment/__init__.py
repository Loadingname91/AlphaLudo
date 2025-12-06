"""
Gymnasium environment registration for Ludo.
"""

from gymnasium.envs.registration import register

register(
    id="Ludo-v0",
    entry_point="rl_agent_ludo.environment.ludoEnv:LudoEnv",
    max_episode_steps=10000,
)

# Unified Ludo Environments (Egocentric Physics approach)
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

