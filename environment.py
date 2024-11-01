from __future__ import annotations

import random
import enum
import itertools
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class Suit(enum.Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


trump: Suit | None = None


class Rank(enum.IntEnum):
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

    def can_beat(self, attack_card: Card) -> bool:
        if self.suit != trump:
            return self.suit == attack_card.suit and self.rank > attack_card.rank

        if attack_card.suit == trump:
            return self.rank > attack_card.rank

        return True


def get_full_deck() -> list[Card]:
    return [Card(suit, rank) for suit, rank in itertools.product(Suit, Rank)]


class Player:
    def __init__(self, game: Game, name: str):
        self.game = game
        self.name = name
        self.cards: list[Card] = []

    def __str__(self):
        return self.name

    @property
    def card_amount(self) -> int:
        if len(self.cards) > 36:
            raise RuntimeError('LOGIC')
        return len(self.cards)

    def take_card(self, card: Card) -> None:
        self.cards.append(card)

    def take_cards(self, cards: list[Card] | None) -> None:
        if cards is None:
            return None
        self.cards += cards

    def attack(self) -> Card:
        if not self.cards:
            raise RuntimeError
        card = random.choice(self.cards)
        return card

    def defend(self, attack_card: Card) -> Card | None:
        choices = self.get_possible_defend_moves(attack_card)
        choice = random.choice(choices)
        return choice

    def remove_card(self, card: Card) -> None:
        self.cards.remove(card)

    def get_possible_defend_moves(self, attack_card: Card) -> list[Card | None]:
        cards: list[Card | None] = []
        for card in self.cards:
            if card.can_beat(attack_card):
                cards.append(card)
        cards.append(None)
        return cards

    def get_possible_attack_moves(self, desk: list[CardPair]) -> list[Card | None]:
        ranks_in_play = set()
        for card in desk:
            ranks_in_play.add(card.attack_card.rank)
            if card.defend_card:
                ranks_in_play.add(card.defend_card.rank)

        cards: list[Card | None] = [
            card for card in self.cards if card.rank in ranks_in_play
        ]
        cards.append(None)
        return cards

    def coattack(self, desk: list[CardPair]) -> Card | None:
        choices = self.get_possible_attack_moves(desk)
        choice = random.choice(choices)

        return choice


@dataclass
class CardPair:
    attack_card: Card
    defend_card: Card | None


class Game:
    def __init__(self) -> None:
        self.deck = get_full_deck()
        self.players: deque[Player] = deque()
        self.winners: list[Player] = []

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def init_game(self) -> None:
        global trump
        random.shuffle(self.deck)
        random.shuffle(self.players)

        for i in range(6):
            for player in self.players:
                if len(self.deck) == 1:
                    trump = self.deck[0].suit

                card = self.deck.pop()
                player.take_card(card)

        if trump is None:
            trump = self.deck[-1].suit

    def is_end_game(self) -> bool:
        return len(self.players) == 1

    def get_attacker(self) -> Player:
        return self.players[0]

    def get_defender(self) -> Player:
        return self.players[1]

    def get_coattacker(self) -> Player | None:
        if len(self.players) > 2:
            return self.players[2]
        return None

    @staticmethod
    def move_loop(
        attacker: Player, defender: Player, coattacker: Player | None
    ) -> list[Card] | None:
        desk: list[CardPair] = []

        attack_card = attacker.attack()

        desk.append(CardPair(attack_card, None))

        defend_card = defender.defend(attack_card)

        while defend_card is not None:
            desk[-1] = CardPair(attack_card, defend_card)
            try:
                defender.remove_card(defend_card)
            except ValueError:
                print(f'{defender.cards=}')
                print(f'{defend_card=}')
                print(f'{str(trump)=}')
                raise

            attack_card = attacker.coattack(desk)

            if attack_card is None:  # нечего подкинуть
                if coattacker is not None:  # есть другие нападающие
                    attack_card = coattacker.coattack(desk)

                    if attack_card is None:  # они не смогли подкинуть
                        break

                    coattacker.remove_card(attack_card)  # они смогли
                    desk.append(CardPair(attack_card, None))
                    defend_card = defender.defend(attack_card)
                else:
                    break

            else:
                attacker.remove_card(attack_card)
                desk.append(CardPair(attack_card, None))
                defend_card = defender.defend(attack_card)

        cards_to_take = [pair.attack_card for pair in desk]

        cards_to_take += [pair.defend_card for pair in desk if pair.defend_card]

        return cards_to_take

    def game_loop(self) -> None:
        while not self.is_end_game():
            # move
            attacker = self.get_attacker()
            defender = self.get_defender()
            coattacker = self.get_coattacker()

            take_cards = self.move_loop(attacker, defender, coattacker)
            logger.error(f"{attacker.card_amount=}")
            logger.error(f"{defender.card_amount=}")
            logger.error(f"{len(self.deck)=}")
            logger.error("=======================")
            # pass
            defender.take_cards(take_cards)

            # take cards up to 6
            for player in self.players:
                while player.card_amount < 6 and self.deck:
                    player.take_card(self.deck.pop())

            # rotate players
            self.players.append(self.players.popleft())  # moving attacker
            if take_cards:
                self.players.append(
                    self.players.popleft()
                )  # moving defender IF they took

            # remove winners
            for player in list(self.players):
                if not player.card_amount:
                    self.winners.append(player)
                    self.players.remove(player)


if __name__ == "__main__":
    for i in range(100):
        game = Game()
        print(len(game.deck))
        player1 = Player(game, "1")
        player2 = Player(game, "2")
        player3 = Player(game, "3")
        game.add_player(player1)
        game.add_player(player2)
        game.add_player(player3)
        game.init_game()
        game.game_loop()
        print(game.winners)
