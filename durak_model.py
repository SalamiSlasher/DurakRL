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


from environment import (
    Card,
    get_possible_attack_moves,
    get_possible_defend_moves,
    CardPair,
    is_possible_attack,
)

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
        hand: list[Card],
        table: list[CardPair],
        discard_pile: list[Card],
        seen_opponent_cards: list[Card],
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

        if len(state.table) == 0:  # no cards in play
            return mask

        for i, card in enumerate(state.hand):
            if not is_possible_attack(card, state.table):
                mask[i + 2] = 0

    else:
        mask[0] = 0  # cannot say Bita

        if not state.table:
            raise AssertionError(
                "Unreachable: cannot defend when there are no cards in play"
            )

        for i, card in enumerate(state.hand):
            attack_cards = []
            for attack_card, defend_card in state.table:
                if defend_card is None:
                    attack_cards.append(attack_card)

            if len(attack_cards) > 1:
                raise AssertionError("Unreachable: more than one attack_card")

            if len(attack_cards) < 1:
                raise AssertionError(
                    "Unreachable: cannot defend when there are no cards in play"
                )

            if not card.can_beat(attack_cards[0]):
                mask[i + 2] = 0  # do not choose from these cards

    return mask
