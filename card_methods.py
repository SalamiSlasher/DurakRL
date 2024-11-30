from __future__ import annotations

import itertools
from enum import Enum, IntEnum
from random import shuffle


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"
    VOID = "void"


class Rank(IntEnum):
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    VOID = 15


class Card:
    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def __repr__(self) -> str:
        return f"{self.rank.name} of {self.suit.name}"


def can_beat(attack_card: Card, defend_card: Card, trump: Suit) -> bool:
    if attack_card.suit == defend_card.suit:
        return defend_card.rank > attack_card.rank
    return defend_card.suit == trump


def get_full_deck() -> list[Card]:
    deck = [Card(suit, rank) for suit, rank in itertools.product(Suit, Rank)]
    shuffle(deck)
    return deck


def get_possible_movements(state: list[Card]) -> list[Card]:
    return []


def possible_attack_cards(
    card_stack: list[Card], player_cards: list[Card]
) -> list[Card]:
    # Собираем ранги всех карт на столе, исключая пустые карты
    table_ranks = set(card.rank for card in card_stack if not is_void_card(card))
    # Возвращаем карты из руки игрока, ранги которых есть на столе
    possible_cards = [card for card in player_cards if card.rank in table_ranks]
    return possible_cards


void_card = Card(Suit.VOID, Rank.VOID)


def is_void_card(card: Card):
    return void_card == card
