import gymnasium as gym
import numpy as np

import rl_agent_ludo.environment.ludoEnv as ludoEnv


def main() -> None:
    """
    Simple smoke test for the Ludo Gymnasium environment.

    Runs a short random rollout and prints step information to stdout.
    Tests rendering functionality as well.
    """
    env = gym.make("Ludo-v0", player_id=0, num_players=4, tokens_per_player=4)

    obs, info = env.reset(seed=2024)
    print("Initial obs (numpy array):", obs)
    print("Initial obs shape:", obs.shape)
    print("Full State available in info['state']:", info["state"])

    # Test render after reset
    print("\n--- Testing render() after reset ---")
    print("Displaying initial board state...")
    try:
        # Use unwrapped to bypass Gymnasium wrapper
        env.unwrapped.render(mode="human")
        print("Board displayed! (Press any key in the OpenCV window to continue, or wait 2 seconds)")
        import time
        time.sleep(2)  # Give user time to see the board
    except Exception as e:
        print(f"Render error: {e}")

    for t in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step {t}: action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"obs_shape={obs.shape}"
        )
        # render every step
        env.unwrapped.render(mode="human")
        input("Press Enter to continue...")

    # Test render after episode ends
    print("\n--- Testing render() after episode ---")
    print("Displaying final board state...")
    try:
        env.unwrapped.render(mode="human")
        print("Final board displayed! (Press any key in the OpenCV window to close, or wait 3 seconds)")
        import time
        time.sleep(3)  # Give user time to see the final state
    except Exception as e:
        print(f"Final render error: {e}")

    # Also test rgb_array mode
    print("\n--- Testing render(mode='rgb_array') ---")
    try:
        img = env.unwrapped.render(mode="rgb_array")
        if img is not None:
            print(f"RGB array render: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    except Exception as e:
        print(f"RGB array render error: {e}")

    env.close()
    print("\nSmoke test completed successfully!")


if __name__ == "__main__":
    main()


