import numpy as np
from game_env import (
    DurakEnv,
    decode_action,
)

# from game_env import get_action_mask

np.random.seed(1)

def test_random_play():
    env = DurakEnv()

    state, info = env.reset()
    action_mask = info["action_mask"]
    print("Initial tensor state:", state)

    done = False
    total_reward = 0

    while not done:
        valid_indices = [i for i, valid in enumerate(action_mask) if valid == 1]
        action_index = np.random.choice(valid_indices)

        state, reward, done, _, info = env.step(action_index)
        action_mask = info["action_mask"]

        print(f"Action: {action_index}")
        print(f"Reward: {reward}")
        print(f"Valid actions: {valid_indices}")
        print("=" * 50)

        total_reward += reward

    print("Game over!")
    print("Total reward:", total_reward)


if __name__ == "__main__":
    test_random_play()