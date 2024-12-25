from __future__ import annotations

from collections import deque
from collections import namedtuple

import random
from typing import NamedTuple

from torch import nn
from torch import optim

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

# If you saved the environment code in a file called durak_env.py, import it:
from durak_env import DurakAEC, DurakObservation

# Otherwise, paste the provided DurakAEC environment code above this snippet and just do:
env = DurakAEC()

# --------------- Replay Buffer ---------------

# Transition = namedtuple(
#     'Transition', ('state', 'action', 'reward', 'next_state', 'done')
# )


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class TransitionTensorTuple(NamedTuple):
    state: torch.Tensor[torch.float32]
    action: torch.Tensor[torch.long]
    reward: torch.Tensor[torch.float32]
    next_state: torch.Tensor[torch.float32]
    done: torch.Tensor[torch.float32]


class ReplayBuffer:
    """
    Simple replay buffer for storing transitions.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> TransitionTensorTuple:
        batch = random.sample(self.buffer, batch_size)

        # Вместо torch.tensor([t.state for t in batch], ...)
        arr_states = np.array([t.state for t in batch], dtype=np.float32)
        states = torch.from_numpy(arr_states)

        arr_next_states = np.array(
            [t.next_state for t in batch], dtype=np.float32
        )
        next_states = torch.from_numpy(arr_next_states)

        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --------------- DQN Network ---------------


class DQNNetwork(nn.Module):
    """
    Example fully-connected DQN using BatchNorm and SELU activation.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        # Feel free to customize the network architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Sequential:
        return self.net(x)


# --------------- Helper Functions ---------------


def flatten_observation(obs_dict: DurakObservation) -> np.array:
    """
    Convert the dictionary observation into a flat np.array.
    Here is an example approach; you might want to alter it
    or do a more sophisticated representation.
    """
    # Example fields in the observation (see env.observation_spaces):
    # hand (36,), cards_on_table (36,), cards_in_discard (36,)
    # trump_suit (1), is_attacker (1), opp_cards_count (1),
    # deck_count (1), attacking_card (1)
    #
    # We will just concatenate them into one 1D array of length 36+36+36+1+1+1+1+1=113.
    # Note: Some fields are scalar, but we'll cast them to array of size 1.
    hand = np.array(obs_dict['hand'] + [0] * (36 - len(obs_dict['hand'])))[:36]
    table = np.array(
        obs_dict['cards_on_table']
        + [0] * (36 - len(obs_dict['cards_on_table']))
    )[:36]
    discard = np.array(
        obs_dict['cards_in_discard']
        + [0] * (36 - len(obs_dict['cards_in_discard']))
    )[:36]
    trump_suit = np.array([obs_dict['trump_suit']], dtype=np.int32)
    is_attacker = np.array([obs_dict['is_attacker']], dtype=np.int32)
    opp_cards_count = np.array([obs_dict['opp_cards_count']], dtype=np.int32)
    deck_count = np.array([obs_dict['deck_count']], dtype=np.int32)
    attacking_card = np.array(
        [
            obs_dict['attacking_card']
            if obs_dict['attacking_card'] is not None
            else -1
        ],
        dtype=np.int32,
    )

    return np.concatenate([
        hand,
        table,
        discard,
        trump_suit,
        is_attacker,
        opp_cards_count,
        deck_count,
        attacking_card,
    ])


def select_action(
    policy_net: DQNNetwork,
    state: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    """
    Epsilon-greedy action selection.
    """
    if random.random() < epsilon:
        # random action
        return random.randint(0, action_dim - 1)
    # greedy action from Q-network
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(torch.argmax(q_values, dim=1).item())


# --------------- Training Loop ---------------


def train_dqn(
    env: DurakAEC,
    num_episodes: int = 5000,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.999,
    target_update_interval: int = 50,
    checkpoint_interval: int = 500,
    save_path: str = 'durak_dqn_final.pth',
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Training on device:', device)

    # Prepare replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    env.reset()
    env.start()
    # We first need to figure out the state_dim, action_dim
    # The action space is Discrete(38).
    action_dim = env.action_spaces[env.agents[0]].n  # 38
    # Flatten an observation to see how large the state is
    temp_obs = flatten_observation(env.observe(env.agents[0]))
    state_dim = temp_obs.shape[0]

    # Create the Q-network and target network
    policy_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    epsilon = epsilon_start
    episode_rewards = []

    # Для подсчёта winrate последнего 100-эпизодного блока
    wins_in_last_100 = 0

    # Simple function to update the Q-network
    def optimize_model():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size
        )

        states = states.to(device)
        actions = actions.to(device).unsqueeze(1)  # shape (batch, 1)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

        # Current Q values
        q_values = policy_net(states).gather(1, actions)  # (batch_size, 1)

        # Next Q values (max over actions) from target_net
        with torch.no_grad():
            max_next_q = (
                target_net(next_states).max(1)[0].unsqueeze(1)
            )  # (batch_size, 1)
            target_q = rewards + (1 - dones) * gamma * max_next_q

        loss = F.mse_loss(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        optimizer.step()

    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        try:
            # Reset the environment
            env.reset()
            env.start()  # Important: call start() to deal the cards etc.

            # In this example, we store transitions from both agents
            # in the same replay buffer using the same policy_net.
            # This is the self-play approach (both agents share weights).

            # Observations
            obs = {}
            for agent in env.agents:
                obs[agent] = flatten_observation(env.observe(agent))

            done = False
            total_reward = 0.0

            while True:
                agent = env.agent_selection

                if env.terminations[agent] or env.truncations[agent]:
                    # If this agent is done, environment should move on
                    env.step(
                        None
                    )  # step with a None action gets skipped by PettingZoo
                else:
                    # Epsilon-greedy action
                    action = select_action(
                        policy_net, obs[agent], epsilon, action_dim, device
                    )
                    # Store current state for the agent
                    state_old = obs[agent]

                    # Step the environment
                    env.step(action)
                    reward = env.rewards[agent]
                    total_reward += reward

                    done_flag = (
                        env.terminations[agent] or env.truncations[agent]
                    )

                    # Next observation
                    obs[agent] = flatten_observation(env.observe(agent))

                    # Add transition to buffer
                    replay_buffer.push(
                        state_old, action, reward, obs[agent], float(done_flag)
                    )

                    # Optimize the model
                    optimize_model()

                # If the env says game done for all agents or some terminal condition, break
                if all(
                    env.terminations[a] or env.truncations[a]
                    for a in env.agents
                ):
                    done = True

                if done:
                    break

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

            # Update target network
            if episode % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            episode_rewards.append(total_reward)

            if env.rewards['player_0'] > 0:
                wins_in_last_100 += 1

            # Save model checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = f'durak_dqn_checkpoint_{episode + 1}.pth'
                torch.save(policy_net.state_dict(), checkpoint_path)
                print(
                    f'[Episode {episode + 1}] Checkpoint saved to {checkpoint_path}'
                )

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                winrate_100 = wins_in_last_100 / 100.0
                print(
                    f'[Episode {episode + 1}] AvgReward(last100)={avg_reward:.3f}, '
                    f'WinRate(last100)={winrate_100:.3f}, Epsilon={epsilon:.3f}'
                )
                wins_in_last_100 = 0  # обнуляем счётчик
        except Exception:
            save_state_path = f'durak_dqn_save_state_{episode + 1}.pth'
            print(f'FAILED. SAVING TO {save_state_path}')
            torch.save(policy_net.state_dict(), save_state_path)

    # Final save
    torch.save(policy_net.state_dict(), save_path)
    print(f'Training finished. Model weights saved to {save_path}')


if __name__ == '__main__':
    # ---------------------------
    # Example usage
    # ---------------------------

    # Paste your DurakAEC code above or import it if in a separate file.
    env = DurakAEC()
    train_dqn(env, num_episodes=5000)
