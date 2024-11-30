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
from game_env import CardLoopStackItem
from game_env import is_possible_attack


TABLE_TENSOR_LEN = 36 * 37
#
INPUT_LEN = (
    36  # hand
    + 36 * 37  # table
    + 36  # bita
    + 1  # is attacker
    + 4  # what suit is trump
)


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
            table_tensor[mapping[card_pair]] = 1

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


Bita: Final[str] = "Bita"
Take: Final[str] = "Take"

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

    def len(self) -> int:
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


def select_action(state: GameState, agent: Actor) -> float:
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    mask = get_action_mask(state)

    if sample > eps_threshold:
        with torch.no_grad():
            q_values = agent(state)  # Прогнозируем значения Q для всех действий
            masked_q_values = q_values * torch.tensor(mask)  # Применяем маску
            return torch.argmax(masked_q_values).item()
    else:
        valid_actions = [i for i, m in enumerate(mask) if m == 1]
        return random.choice(valid_actions)  # Выбираем случайное действие


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    def __init__(self, start_hand: list[Card], trump: Suit) -> None:
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
