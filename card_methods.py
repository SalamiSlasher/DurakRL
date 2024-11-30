from __future__ import annotations

import itertools
from enum import Enum, IntEnum
from random import shuffle


class Suit(Enum):
    HEARTS = "♡"
    DIAMONDS = "♢"
    CLUBS = "♣"
    SPADES = "♠"
    VOID = "void"

    def __int__(self):
        if self == Suit.VOID:
            return 0
        elif self == Suit.HEARTS:
            return 1
        elif self == Suit.DIAMONDS:
            return 2
        elif self == Suit.CLUBS:
            return 3
        elif self == Suit.SPADES:
            return 4



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
    encoder = {
        Rank.SIX: '6',
        Rank.SEVEN: '7',
        Rank.EIGHT: '8',
        Rank.NINE: '9',
        Rank.TEN: '10',
        Rank.JACK : 'J',
        Rank.QUEEN : 'Q',
        Rank.KING : 'K',
        Rank.ACE : 'A',
        Rank.VOID : 'VOID',
    }

    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __repr__(self) -> str:
        return f"{self.encoder[self.rank]}{self.suit.value}"

    @property
    def id(self) -> int:
        """
        Возвращает уникальный идентификатор карты.
        Идентификатор - это целое число, которое зависит от масти и достоинства.
        """
        return (self.rank.value - 6) + (int(self.suit.value) * 9)

    @staticmethod
    def can_beat(attack_card: Card, defend_card: Card, trump: Suit) -> bool:
        if attack_card.suit == defend_card.suit:
            return defend_card.rank > attack_card.rank
        return defend_card.suit == trump and attack_card.suit != trump

    def can_this_beat(self, attack_card: Card, trump: Suit) -> bool:
        return Card.can_beat(attack_card, self, trump)

def get_full_deck() -> list[Card]:
    deck = [Card(suit, rank) for suit, rank in itertools.product(Suit, Rank) if not is_void_card(Card(suit, rank))]
    shuffle(deck)
    return deck


def get_possible_movements(state: list[Card]) -> list[Card]:
    return []


def possible_attack_cards(
    card_stack: list[Card], player_cards: list[Card]
) -> list[Card]:
    if len(card_stack) == 0:
        return player_cards

    # Собираем ранги всех карт на столе, исключая пустые карты
    table_ranks = set(card.rank for card in card_stack if not is_void_card(card))
    # Возвращаем карты из руки игрока, ранги которых есть на столе
    possible_cards = [card for card in player_cards if card.rank in table_ranks]
    return possible_cards


def possible_defend_cards(attack_card: Card, player_cards: list[Card], trump: Suit) -> list[Card]:
    # Фильтруем карты игрока, которые могут побить атакующую карту
    defend_cards = [card for card in player_cards if Card.can_beat(attack_card, card, trump)]
    defend_cards.append(void_card)
    return defend_cards


void_card = Card(Suit.VOID, Rank.VOID)


def is_void_card(card: Card):
    return card.rank == Rank.VOID or card.suit == Suit.VOID
