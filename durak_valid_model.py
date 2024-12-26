from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Импортируем вашу среду DurakAEC,
# а также вспомогательные функции card_can_beat, is_rank_in_play
from durak_env import DurakAEC, DurakObservation, card_can_beat, is_rank_in_play

# ========== 1. valid_actions ==========


def valid_actions(env: DurakAEC, agent: str) -> list[int]:
    """
    Возвращает список допустимых (валидных) действий для данного агента.
    Логика соответствует вашему DurakAEC.step(...).
    """
    if env.terminations[agent] or env.truncations[agent]:
        return []

    hand = env.hands[agent]  # множество карт в руке
    is_attacker = env.is_attacker[agent]
    attacking_card = env.attacking_card

    va = []
    if is_attacker:
        # если стол пуст => любые карты из руки
        # иначе => только те, чей rank есть на столе
        if len(env.cards_on_table) == 0:
            va.extend(list(hand))
        else:
            for c in hand:
                if is_rank_in_play(c, env.cards_on_table):
                    va.append(c)
            va.append(37)
            # "бито" (37) для атакующего
            # "взять" (36) обычно не имеет смысла для атакующего => не добавляем
    else:
        # Защитник
        if attacking_card is not None:
            for c in hand:
                if card_can_beat(c, env.trump_suit, attacking_card):
                    va.append(c)
        # "взять" (36) всегда разрешено защитнику
        va.append(36)
        # "бито" (37) в вашем DurakAEC обычно неразрешено защитнику => не добавляем
    return va


# ========== 2. Replay Buffer ==========

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    """
    Simple replay buffer for storing transitions.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        arr_states = np.array([t.state for t in batch], dtype=np.float32)
        arr_next_states = np.array(
            [t.next_state for t in batch], dtype=np.float32
        )
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        states = torch.from_numpy(arr_states)
        next_states = torch.from_numpy(arr_next_states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ========== 3. Сеть ==========


class DQNNetwork(nn.Module):
    """
    Fully-connected сеть (Q-сеть).
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ========== 4. flatten_observation ==========


def flatten_observation(obs_dict: DurakObservation) -> np.ndarray:
    """
    Превращаем Dict-наблюдение DurakObservation в вектор float32 размера [113].
    """
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
    ]).astype(np.float32)


# ========== 5. Softmax-эксперимент ==========


def masked_select_action_softmax(
    policy_net: DQNNetwork,
    env: DurakAEC,
    agent: str,
    obs_state: np.ndarray,
    action_dim: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[int | None, float]:
    """
    1) Смотрим valid_actions(env, agent).
    2) Получаем логиты = Q(s,a) для всех a=0..action_dim-1.
    3) Ставим -1e9 для невалидных.
    4) Применяем Softmax(logits / temperature).
    5) Сэмплим действие из полученного распределения.
    6) Возвращаем (chosen_action, probability_of_chosen_action).

    Если valid_actions пуст, возвращаем (None, 0.0).
    """
    va = valid_actions(env, agent)
    if not va:
        return None, 0.0

    # получаем Q
    state_t = torch.tensor(
        obs_state, dtype=torch.float32, device=device
    ).unsqueeze(0)
    was_training = policy_net.training
    policy_net.eval()
    with torch.no_grad():
        q_values = policy_net(state_t)  # shape=[1, action_dim]
    if was_training:
        policy_net.train()

    # убираем batch-измерение
    q_values = q_values.squeeze(0)  # shape=[action_dim]

    # маскируем
    big_neg = -1e9
    masked_logits = torch.full(
        (action_dim,), fill_value=big_neg, dtype=torch.float32, device=device
    )
    for a in va:
        masked_logits[a] = q_values[a]

    # Превращаем логиты в распределение (Softmax).
    # Можно добавить деление на temperature, чтобы управлять «остротой» распределения.
    dist = torch.distributions.Categorical(logits=masked_logits / temperature)

    # Сэмплим действие
    action_tensor = dist.sample()  # это int-тензор
    action_int = action_tensor.item()

    # Вероятность выбранного действия
    action_prob = dist.probs[action_tensor].item()

    return dist.logits


# ========== 6. Цикл обучения с softmax-политикой ==========


def train_dqn(
    env: DurakAEC,
    num_episodes: int = 5000,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    temperature: float = 1.0,
    temperature_decay: float = 0.999,
    temperature_min: float = 0.1,
    target_update_interval: int = 50,
    checkpoint_interval: int = 500,
    save_path: str = 'durak_dqn_final.pth',
) -> None:
    """
    Пример обучения, где при выборе действия
    мы используем softmax по Q, а не eps-greedy.

    temperature отвечает за «расплывчатость» softmax.
    Медленно уменьшаем её до temperature_min.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on device:', device)

    # 6.1: Инициализация
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    env.reset()
    env.start()
    action_dim = env.action_spaces[env.agents[0]].n  # 38
    sample_obs = flatten_observation(env.observe(env.agents[0]))
    state_dim = sample_obs.shape[0]  # 113

    policy_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    episode_rewards = []
    wins_in_last_100 = 0

    # функция оптимизации
    def optimize_model():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size
        )

        states = states.to(device)
        actions = actions.to(device).unsqueeze(1)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

        # Q(s,a)
        q_current = policy_net(states).gather(1, actions)

        with torch.no_grad():
            q_next_max = target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + (1 - dones) * gamma * q_next_max

        loss = F.mse_loss(q_current, q_target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        optimizer.step()

    # 6.2: Основной цикл эпизодов
    for episode in tqdm(range(num_episodes)):
        env.reset()
        env.start()

        obs = {}
        for ag in env.agents:
            obs[ag] = flatten_observation(env.observe(ag))

        done = False
        total_reward = 0.0

        while not done:
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                # Выбираем действие через softmax
                action = masked_select_action_softmax(
                    policy_net,
                    env,
                    agent,
                    obs[agent],
                    action_dim,
                    device,
                    temperature=temperature,
                )
                if action[0] is None:  # go
                    continue  # means player won
                action = torch.argmax(action).item()

                # try:
                #     action = torch.argmax(action).item()
                # except TypeError:
                #     print(action)
                #     print(type(action))

                if action is None:
                    env.step(None)
                else:
                    state_old = obs[agent].copy()
                    env.step(action)

                    r = env.rewards[agent]
                    total_reward += r
                    done_flag = (
                        env.terminations[agent] or env.truncations[agent]
                    )

                    obs[agent] = flatten_observation(env.observe(agent))

                    # Сохраняем переход
                    replay_buffer.push(
                        state_old, action, r, obs[agent], float(done_flag)
                    )

                    # Оптимизация
                    optimize_model()

            env.render()

            if all(
                env.terminations[a] or env.truncations[a] for a in env.agents
            ):
                done = True

        # итоги эпизода
        episode_rewards.append(total_reward)
        if env.rewards['player_0'] > 0:
            wins_in_last_100 += 1

        # обновляем target
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # статистика
        if (episode + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            winrate_100 = wins_in_last_100 / 100
            print(
                f'[Ep {episode + 1}] AvgR={avg_r:.2f}, WinRate_100={winrate_100:.2f}, T={temperature:.3f}'
            )
            wins_in_last_100 = 0

        # checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = f'durak_dqn_ckpt_{episode + 1}.pth'
            torch.save(policy_net.state_dict(), ckpt_path)
            print(f'Saved checkpoint to {ckpt_path}')

        # плавно уменьшаем temperature
        temperature = max(temperature_min, temperature_decay * temperature)

    # финальная модель
    torch.save(policy_net.state_dict(), save_path)
    print(f'Done. Model saved to {save_path}')


# ========== 7. Пример запуска ==========

if __name__ == '__main__':
    env = DurakAEC()
    train_dqn(env, num_episodes=2000, checkpoint_interval=100)
