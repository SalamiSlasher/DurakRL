from __future__ import annotations

from typing import NamedTuple, Final, Protocol, TypeAlias

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from card_methods import Suit, void_card, mapping

from card_methods import Card
from game_env import CardLoopStackItem, Bita, Take
from game_env import is_possible_attack


TABLE_TENSOR_LEN = 36 * 37

INPUT_LEN = (
    36  # hand
    + 36 * 37  # table
    + 36  # bita
    + 1  # is attacker
    + 4  # what suit is trump
)

OUTPUT_LEN = 38


plt.ion()


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class GameState:
    def __init__(
        self,
        hand: list[Card],
        table: list[CardLoopStackItem],
        discard_pile: list[Card],
        seen_opponent_cards: list[Card],
        is_attacker: bool,
        trump: Suit,
    ) -> None:
        self.hand = hand
        self.table = table
        self.discard_pile = discard_pile
        self.seen_opponent_cards = seen_opponent_cards
        self.is_attacker = is_attacker
        self.trump = trump

    def to_tensor(self) -> torch.Tensor:
        hand_tensor = torch.zeros(36)  # Например, 36 карт в колоде
        for card in self.hand:
            hand_tensor[card.id] = 1

        table_tensor = torch.zeros(
            TABLE_TENSOR_LEN
        )  # Для каждой пары карты (атака + защита)
        for card_pair in self.table:
            idx = mapping[(card_pair.attacker_card, card_pair.defender_card)]
            table_tensor[idx] = 1

        discard_pile_tensor = torch.zeros(36)  # Битая колода
        for card in self.discard_pile:
            discard_pile_tensor[card.id] = 1

        # seen_opponent_cards_tensor = torch.zeros(
        #     36
        # )  # Карты противника, которые мы видели
        # for card in self.seen_opponent_cards:
        #     seen_opponent_cards_tensor[card.id] = 1

        is_attacker_tensor = torch.tensor([1.0 if self.is_attacker else 0.0])
        trump_tensor = torch.zeros(4)
        trump_tensor[int(self.trump)] = 1

        # Объединяем все тензоры в один вектор
        return torch.cat(
            [
                hand_tensor,
                table_tensor,
                discard_pile_tensor,
                # seen_opponent_cards_tensor,
                is_attacker_tensor,
                trump_tensor,
            ]
        )


Action: TypeAlias = Card | Bita | Take


class Transition(NamedTuple):
    state: GameState
    action: Action
    next_state: GameState
    reward: float
    done: bool


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


def get_attack_action_mask(mask: list[int], state: GameState) -> list[int]:
    mask[1] = 0  # cannot say Take

    if len(state.table) == 0:  # no cards in play
        return mask

    for i, card in enumerate(state.hand):
        if not is_possible_attack(card, state.table):
            mask[i + 2] = 0

    return mask


def get_defender_action_mask(mask: list[int], state: GameState) -> list[int]:
    mask[0] = 0  # cannot say Bita

    if not state.table:
        raise AssertionError(
            "Unreachable: cannot defend when there are no cards in play"
        )

    for i, card in enumerate(state.hand):
        attack_cards = []
        for card_pair in state.table:
            if card_pair.defender_card == void_card:
                attack_cards.append(card_pair.attacker_card)

        if len(attack_cards) > 1:
            raise AssertionError("Unreachable: more than one attack_card")

        if len(attack_cards) < 1:
            raise AssertionError(
                "Unreachable: cannot defend when there are no cards in play"
            )

        if not Card.can_beat(
            attack_card=attack_cards[0], defend_card=card, trump=state.trump
        ):
            mask[i + 2] = 0  # do not choose from these cards

    return mask


def get_action_mask(state: GameState) -> list[int]:
    mask = [1] * (len(state.hand) + 2)
    # mask[0] == Bita  # first action in mark is always Bita
    # mask[1] == Take  # second action in mark is always Take
    # isinstance(mask[2:], list[Card]) == True  # rest elements are Card objects

    if state.is_attacker:
        mask = get_attack_action_mask(mask, state)

    else:
        mask = get_defender_action_mask(mask, state)

    return mask


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
    def __init__(self, n_observations: int = INPUT_LEN, n_actions: int = OUTPUT_LEN):
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
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Инициализация целевой сети
        self.target_net.eval()  # Остановка обновлений target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state: GameState) -> int:
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
                q_values = self.policy_net(state.to_tensor())  # Получаем прогноз Q-значений
                masked_q_values = q_values * torch.tensor(mask)  # Применяем маску
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

        state_batch = torch.stack([transition.state.to_tensor() for transition in transitions])
        action_batch = torch.tensor([transition.action for transition in transitions]).unsqueeze(1)
        reward_batch = torch.tensor([transition.reward for transition in transitions])
        next_state_batch = torch.stack([transition.next_state.to_tensor() for transition in transitions])
        done_batch = torch.tensor([transition.done for transition in transitions])

        # Получаем Q-значения для текущих состояний
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Получаем максимальные Q-значения для будущих состояний из целевой сети
        next_state_values = self.target_net(next_state_batch).max(1)[0]
        next_state_values = next_state_values.detach()

        # Рассчитываем целевые значения для Q
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

        # Рассчитываем потери
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        # Обновляем веса сети
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Ограничиваем градиенты
        self.optimizer.step()

    def update_target_network(self):
        """
        Обновляем целевую сеть
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


def play_game(agent: Agent, environment: GameEnvironment) -> float:
    """
    Функция для игры одного эпизода
    """
    state = environment.reset()  # Инициализация состояния игры
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state)  # Выбор действия
        next_state, reward, done = environment.step(action)  # Выполнение действия и получение нового состояния
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


if __name__ == "__main__":
    random.seed(42)
