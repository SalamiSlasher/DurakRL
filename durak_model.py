from __future__ import annotations

from typing import Final
from typing import NamedTuple
from typing import Protocol
from typing import TypeAlias

from collections import deque
from itertools import count

import math
import random

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from card_methods import Card
from card_methods import Suit
from card_methods import mapping
from card_methods import void_card
from game_env import INPUT_LEN
from game_env import OUTPUT_LEN
from game_env import Bita
from game_env import CardLoopStackItem
from game_env import DurakEnv
from game_env import GameState
from game_env import Take
from game_env import Transition
from game_env import get_action_mask
from game_env import is_possible_attack

plt.ion()


device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Actor(Protocol):
    def __call__(self, state: GameState) -> torch.Tensor: ...

    policy_net: DQN


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
steps_done = 0


class DQN(nn.Module):
    def __init__(
        self, n_observations: int = INPUT_LEN, n_actions: int = OUTPUT_LEN
    ):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))

        logits = self.layer3(x)

        probabilities = F.softmax(logits, dim=-1)

        return probabilities


class Agent:
    def __init__(self, start_hand: list[Card], trump: Suit) -> None:
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )  # Инициализация целевой сети
        self.target_net.eval()  # Остановка обновлений target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state: GameState) -> float:
        """
        Выбор действия на основе ε-жадности
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        mask = get_action_mask(state)

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(
                    state.to_tensor()
                )  # Получаем прогноз Q-значений
                masked_q_values = q_values * torch.tensor(
                    mask
                )  # Применяем маску
                return torch.argmax(masked_q_values).item()
        else:
            valid_actions = [i for i, m in enumerate(mask) if m == 1]
            return random.choice(valid_actions)  # Выбираем случайное действие

    def optimize_model(self) -> None:
        """
        Обучение модели через минимизацию ошибки Q-функции
        """
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        state_batch = torch.stack([
            transition.state.to_tensor() for transition in transitions
        ])
        action_batch = torch.tensor([
            transition.action for transition in transitions
        ]).unsqueeze(1)
        reward_batch = torch.tensor([
            transition.reward for transition in transitions
        ])
        next_state_batch = torch.stack([
            transition.next_state.to_tensor() for transition in transitions
        ])
        done_batch = torch.tensor([
            transition.done for transition in transitions
        ])

        # Получаем Q-значения для текущих состояний
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # Получаем максимальные Q-значения для будущих состояний из целевой сети
        next_state_values = self.target_net(next_state_batch).max(1)[0]
        next_state_values = next_state_values.detach()

        # Рассчитываем целевые значения для Q
        expected_state_action_values = reward_batch + (
            GAMMA * next_state_values * (1 - done_batch)
        )

        # Рассчитываем потери
        loss = F.mse_loss(
            state_action_values.squeeze(), expected_state_action_values
        )

        # Обновляем веса сети
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad:
                param.grad.data.clamp_(-1, 1)  # Ограничиваем градиенты
        self.optimizer.step()

    def update_target_network(self):
        """
        Обновляем целевую сеть
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


TARGET_UPDATE = 100


def play_game(agent: Agent, environment: DurakEnv) -> float:
    """
    Функция для игры одного эпизода
    """
    state = environment.reset()  # Инициализация состояния игры
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state)  # Выбор действия
        next_state, reward, done = environment.step(
            action
        )  # Выполнение действия и получение нового состояния
        total_reward += reward

        # Сохранение перехода в буфер
        transition = Transition(state, action, next_state, reward, done)
        agent.memory.push(transition)

        # Обучение модели
        agent.optimize_model()

        # Обновление целевой сети с определенной частотой
        if agent.steps_done % TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.steps_done += 1

        state = next_state

    return total_reward


if __name__ == '__main__':
    random.seed(42)
