"""
Example: Creating and Using a Custom Agent

This example demonstrates how to:
1. Create a custom agent class
2. Register it with AgentRegistry
3. Use it in training

Run this example:
    python examples/custom_agent_example.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from rl_agent_ludo.agents.base_agent import Agent
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from rl_agent_ludo.utils.state import State
from rl_agent_ludo.environment.ludo_env import LudoEnv
from typing import Optional
import random


class SimpleGreedyAgent(Agent):
    """
    A simple greedy agent that always moves the piece closest to goal.
    
    This is a minimal example showing how to create a custom agent.
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize the greedy agent."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    @property
    def is_on_policy(self) -> bool:
        """This agent doesn't learn, so it's off-policy."""
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """This agent doesn't learn."""
        return False
    
    def act(self, state: State) -> int:
        """
        Select action by moving the piece closest to goal.
        
        If multiple pieces are equally close, pick randomly.
        """
        if not state.valid_moves:
            return 0
        
        # Score each valid move by piece position (higher = closer to goal)
        scores = {}
        for action_idx in state.valid_moves:
            # Get the actual piece index
            if state.movable_pieces and action_idx < len(state.movable_pieces):
                piece_idx = state.movable_pieces[action_idx]
            else:
                piece_idx = action_idx
            
            # Score: piece position (higher is better, goal is 57)
            piece_pos = state.player_pieces[piece_idx]
            scores[action_idx] = piece_pos
        
        # Find best score
        best_score = max(scores.values())
        best_actions = [a for a, s in scores.items() if s == best_score]
        
        # If tie, pick randomly
        return random.choice(best_actions)


def main():
    """Example: Register and use a custom agent."""
    
    print("=" * 60)
    print("Custom Agent Example")
    print("=" * 60)
    
    # Step 1: Register the custom agent
    print("\n1. Registering custom agent...")
    AgentRegistry.register_agent('simple_greedy', SimpleGreedyAgent)
    print("   ✓ Registered 'simple_greedy' agent")
    
    # Step 2: Verify it's available
    available = AgentRegistry.get_available_agents()
    print(f"\n2. Available agents: {available}")
    assert 'simple_greedy' in available, "Agent should be registered"
    
    # Step 3: Create agent from config
    print("\n3. Creating agent from config...")
    config = {
        'type': 'simple_greedy',
        'seed': 42
    }
    agent = AgentRegistry.create_agent(config)
    print(f"   ✓ Created agent: {type(agent).__name__}")
    
    # Step 4: Test agent in environment
    print("\n4. Testing agent in environment...")
    env = LudoEnv(seed=42)
    state = env.reset()
    
    print(f"   Initial state:")
    print(f"   - Valid moves: {state.valid_moves}")
    print(f"   - Player pieces: {state.player_pieces}")
    print(f"   - Dice roll: {state.dice_roll}")
    
    action = agent.act(state)
    print(f"\n   Selected action: {action}")
    print(f"   ✓ Action is valid: {action in state.valid_moves}")
    
    # Step 5: Run a few steps
    print("\n5. Running a few steps...")
    for step in range(5):
        if env.done:
            break
        
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        
        if info.get('is_learning_agent_turn', False):
            print(f"   Step {step + 1}: action={action}, reward={reward:.1f}, done={done}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nTo use this agent in a config file:")
    print("  agent:")
    print("    type: 'simple_greedy'")
    print("    seed: 42")


if __name__ == '__main__':
    main()

