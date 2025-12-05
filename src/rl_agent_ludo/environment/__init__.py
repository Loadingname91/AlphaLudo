"""
Gymnasium environment registration for Ludo.
"""

from gymnasium.envs.registration import register

register(
    id="Ludo-v0",
    entry_point="rl_agent_ludo.environment.ludoEnv:LudoEnv",
    max_episode_steps=10000,
)

