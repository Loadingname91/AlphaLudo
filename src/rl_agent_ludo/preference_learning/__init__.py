"""
Preference Learning Module for T-REX (Level 6).

This module implements preference-based reward learning from ranked trajectories.

Components:
- trajectory_collector: Collect game trajectories from agents
- trajectory_ranker: Create preference pairs from trajectories
- reward_network: Learn reward function from preferences
"""

from .trajectory_collector import TrajectoryCollector

__all__ = ['TrajectoryCollector']
