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

# Импортируем вашу среду.
# Предполагается, что внутри durak_env.py определены:
# class DurakAEC, функции card_can_beat, is_rank_in_play, etc.
from durak_env import DurakAEC, DurakObservation, card_can_beat, is_rank_in_play


# ========== 1. valid_actions ==========


def valid_actions(env: DurakAEC, agent: str) -> list[int]:
    """
    Возвращает список допустимых (валидных) действий для данного агента:
    - Если агент терминальный, возвращаем [].
    - Если это атакующий:
        * можно сыграть любую карту из руки (0..35),
          если стол пуст или ранг этой карты уже есть на столе (is_rank_in_play).
        * можно "бито" (37).
        * не включаем "взять" (36) для атакующего,
          потому что в DurakAEC он обычно игнорируется (нет смысла).
    - Если это защитник:
        * можно сыграть карту, которая бьёт attacking_card,
        * можно "взять" (36),
        * обычно "бито" (37) для защитника не разрешено,
          но если у вас в step(...) разрешено — добавьте.
    """
    if env.terminations[agent] or env.truncations[agent]:
        return []

    hand = env.hands[agent]  # множество карт (int) в руке
    is_attacker = env.is_attacker[agent]
    attacking_card = env.attacking_card

    va = []
    if is_attacker:
        # можно подкинуть любую карту, которая
        #  если стол пуст, нет ограничения
        #  если на столе что-то лежит, проверяем is_rank_in_play
        if len(env.cards_on_table) == 0:
            # стол пуст => любые карты из руки
            va.extend(list(hand))
        else:
            # стол не пуст => только карты, у которых rank есть на столе
            for c in hand:
                if is_rank_in_play(c, env.cards_on_table):
                    va.append(c)

        # "бито" (37) для атакующего
        va.append(37)
        # "взять" (36) мы не добавляем, т.к. по логике DurakAEC step игнорирует
    else:
        # Защитник
        if attacking_card is not None:
            # можно сыграть любую карту из руки, которая бьёт attacking_card
            for c in hand:
                if card_can_beat(
                    defend_card=c,
                    trump_suit=env.trump_suit,
                    attacking_card=attacking_card,
                ):
                    va.append(c)
        # можно "взять" (36)
        va.append(36)
        # "бито" (37) для защитника в вашем коде DurakAEC обычно "Invalid",
        # так что не добавляем. Если хотите разрешить — добавьте va.append(37).

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
        # args => (state, action, reward, next_state, done)
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        # Массово превращаем в numpy
        arr_states = np.array([t.state for t in batch], dtype=np.float32)
        arr_next_states = np.array(
            [t.next_state for t in batch], dtype=np.float32
        )
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        # Теперь в тензоры
        states = torch.from_numpy(arr_states)
        next_states = torch.from_numpy(arr_next_states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ========== 3. Сеть DQN ==========


class DQNNetwork(nn.Module):
    """
    Пример fully-connected DQN, использующий LayerNorm + SELU.
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


# ========== 4. Функция flatten_observation ==========


def flatten_observation(obs_dict: DurakObservation) -> np.ndarray:
    """
    Сжимаем Dict-наблюдение в плоский numpy-вектор (размер 113).
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
    ])


# ========== 5. Выбор действия с маской ==========


def masked_select_action(
    policy_net: DQNNetwork,
    env: DurakAEC,
    agent: str,
    obs_state: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int | None:
    """
    Выбор действия с учётом valid_actions.

    - Если random.random() < epsilon => берём **случайное** действие
      из valid_actions.
    - Иначе => считаем Q для всех действий [0..action_dim-1],
      зануляем/занижаем Q для невалидных, берём argmax.

    Если вообще нет valid_actions => возвращаем None (пропуск хода).
    """

    va = valid_actions(env, agent)  # список допустимых действий
    if not va:
        return None  # нет ходов => сделаем step(None)

    # Если eps => случайное разрешённое действие
    if random.random() < epsilon:
        return random.choice(va)
    else:
        # Вычисляем Q(s, a) для всех a
        state_t = torch.tensor(
            obs_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        # Отключаем BatchNorm/LN обновление статистик (eval mode),
        # хотя LayerNorm менее критичен, всё же на 1 выборке не проблема.
        was_training = policy_net.training
        policy_net.eval()
        with torch.no_grad():
            q_values = policy_net(state_t)  # shape=[1, action_dim]
        if was_training:
            policy_net.train()

        q_values = q_values.squeeze(0).cpu().numpy()  # shape=[action_dim]

        # Маскируем: для недопустимых действий ставим -1e9
        masked_q = np.full(action_dim, -1e9, dtype=np.float32)
        for a in va:
            masked_q[a] = q_values[a]

        # Выбираем argmax
        best_action = int(np.argmax(masked_q))
        return best_action


# ========== 6. Основной цикл обучения ==========


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

    # ----- 6.1 Создаём буфер и сеть -----
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Выясняем размеры
    env.reset()
    env.start()  # первая раздача
    action_dim = env.action_spaces[env.agents[0]].n  # 38
    tmp_obs = flatten_observation(env.observe(env.agents[0]))
    state_dim = tmp_obs.shape[0]  # 113

    policy_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    epsilon = epsilon_start
    episode_rewards = []
    wins_in_last_100 = 0

    # ----- 6.2 Функция оптимизации -----
    def optimize_model():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size
        )

        states = states.to(device)
        actions = actions.to(device).unsqueeze(1)  # (batch,1)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

        # Q(s,a)
        q_values = policy_net(states).gather(1, actions)

        # max Q(s', a') из target_net
        with torch.no_grad():
            max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * gamma * max_next_q

        loss = F.mse_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        optimizer.step()

    # ----- 6.3 Запуск эпизодов -----
    for episode in tqdm(range(num_episodes)):
        env.reset()
        env.start()

        # Собираем начальные obs
        obs = {}
        for ag in env.agents:
            obs[ag] = flatten_observation(env.observe(ag))

        done = False
        total_reward = 0.0

        while not done:
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                # агент уже выбыл, пропускаем
                env.step(None)
            else:
                # Выбираем действие с учётом валидных
                action = masked_select_action(
                    policy_net,
                    env,
                    agent,
                    obs[agent],
                    epsilon,
                    action_dim,
                    device,
                )
                if action is None:
                    # нет валидных ходов => step(None)
                    env.step(None)
                else:
                    state_old = obs[agent].copy()
                    env.step(action)

                    reward = env.rewards[agent]
                    total_reward += reward
                    done_flag = (
                        env.terminations[agent] or env.truncations[agent]
                    )

                    # Новое наблюдение
                    obs[agent] = flatten_observation(env.observe(agent))

                    # Сохраняем в буфер
                    replay_buffer.push(
                        state_old, action, reward, obs[agent], float(done_flag)
                    )
                    # Обновляем политику
                    optimize_model()

            # Проверяем глобальное завершение
            if all(
                env.terminations[a] or env.truncations[a] for a in env.agents
            ):
                done = True

        # Эпизод закончен
        episode_rewards.append(total_reward)

        # Считаем, что если reward у player_0 > 0 => победа player_0
        if env.rewards['player_0'] > 0:
            wins_in_last_100 += 1

        # Апдейтим target
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Каждые 100 эпизодов печатаем статистику
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            winrate_100 = wins_in_last_100 / 100
            print(
                f'[Ep {episode + 1}] AvgReward(last100)={avg_reward:.2f}, '
                f'WinRate(last100)={winrate_100:.2f}, Eps={epsilon:.3f}'
            )
            wins_in_last_100 = 0

        # Сохраняем checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f'durak_dqn_checkpoint_{episode + 1}.pth'
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f'... checkpoint saved to {checkpoint_path}')

        # Декремент eps
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Сохраняем финальную модель
    torch.save(policy_net.state_dict(), save_path)
    print(f'Training finished. Model weights saved to {save_path}')


# ========== 7. Пример запуска ==========

if __name__ == '__main__':
    env = DurakAEC()
    train_dqn(env, num_episodes=5000)
