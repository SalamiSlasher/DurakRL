import itertools
from enum import Enum, IntEnum
from random import shuffle


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"

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

class Card:
    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def __repr__(self) -> str:
        return f"{self.rank.name} of {self.suit.name}"


def get_full_deck() -> list[Card]:
    deck = [Card(suit, rank) for suit, rank in itertools.product(Suit, Rank)]
    shuffle(deck)
    return deck


def get_possible_movements(state: list[Card]) -> list[Card]:
    return []
