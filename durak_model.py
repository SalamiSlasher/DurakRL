from __future__ import annotations

from typing import NamedTuple, Final

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


from environment import Card

env = gym.make("CartPole-v1")

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
        hand: set[Card],
        table: set[Card],
        discard_pile: set[Card],
        seen_opponent_cards: set[Card],
        is_attacker: bool,
    ) -> None:
        self.hand = hand
        self.table = table
        self.discard_pile = discard_pile
        self.seen_opponent_cards = seen_opponent_cards
        self.is_attacker = is_attacker


Bita: Final[str] = "Bita"
Take: Final[str] = "Take"

type Action = Card | Bita | Take


class Transition(NamedTuple):
    state: GameState
    action: Action
    next_state: GameState
    reward: float
    done: bool


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def len(self):
        return len(self.memory)


def get_action_mask(state: GameState) -> list[int]:
    mask = [1] * (len(state.hand) + 2)
    # mask[0] == Bita  # first action in mark is always Bita
    # mask[1] == Take  # second action in mark is always Take
    # isinstance(mask[2:], list[Card]) == True  # rest elements are Card objects

    if state.is_attacker:
        mask[1] = 0  # cannot say Take
    else:
        mask[0] = 0  # cannot say Bita

    if not state.is_attacker and len(state.table) > 0 and len(state.hand) == 0:
        mask = [0] * len(mask)
        mask[1] = 1  # always say Take

    for i, card in enumerate(state.hand):
        if not possible_atack_cards(card, state.table):
            mask[i + 2] = 0  # Карты начинаются с индекса 2

    return mask
