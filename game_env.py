from __future__ import annotations
import random

from gymnasium import Env
from collections import deque

from card_methods import Card, Suit, void_card, possible_attack_cards
import card_methods


class DurakEnv(Env): ...


class DurakGame:
    def __init__(self) -> None:
        # self.player_count = player_count
        # self.players = deque(Player() for i in range(player_count))
        self.trump: Suit | None = None
        self.players = deque(maxlen=6)
        self.deck: list[Card] = card_methods.get_full_deck()
        self.beat: list[Card] = []

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def start_game(self) -> None:
        # --------------INIT--------------
        if len(self.players) < 2:
            raise AssertionError("Not enough players")

        for _ in range(6):
            for player in self.players:
                player.get_cards(self.deck)

        self.trump: Suit = self.deck[0].suit
        Player.trump_suit = self.trump
        # --------------INIT--------------

        while self.end_condition():
            attacker = self.players[0]
            defender = self.players[1]
            co_attacker = None if len(self.players) == 2 else self.players[-1]

            turn_stack = []
            is_beaten_off = self.attack_loop(
                attacker, defender, co_attacker, turn_stack
            )

    def attack_loop(
        self,
        attacker,
        defender,
        co_attacker,
        turn_stack: list[CardLoopStackItem],
        is_beaten_off=True,
    ) -> None:
        if len(defender) - len(turn_stack) == 0:
            return is_beaten_off

        if not attacker.want_to_attack(turn_stack):
            if co_attacker and co_attacker.want_to_attack(turn_stack):
                return self.attack_loop(co_attacker, defender, attacker, turn_stack)
            return is_beaten_off

        # last card in turn_stack is card that must be beated by defender
        attack_card = attacker.attack(turn_stack)

        # Когда дефендер отбил на прошлом ходу
        if is_beaten_off:
            defend_card = defender.defend(turn_stack)
        # Когда дефендер не отбил в прошлом ходе и его топят...
        else:
            defend_card = void_card

        turn_stack_item = CardLoopStackItem(
            attack_card, attacker, defend_card, defender
        )
        turn_stack.append(turn_stack_item)

        is_beaten_off = defend_card != void_card  # void_card - не смог отбить
        self.attack_loop(attacker, defender, co_attacker, turn_stack, is_beaten_off)

    def end_condition(self) -> bool:
        return self.players == 1

    def give_card(self, player: Player) -> None:
        player.get_card(self.deck.pop())


class CardLoopStackItem:
    def __init__(
        self,
        attacker_card: Card,
        attacker: Player,
        defender_card: Card,
        defender: Player,
    ):
        self.attacker_card = attacker_card
        self.attacker = attacker

        self.defender_card: Card = defender_card
        self.defender = defender

    def flatten(self):
        return [self.attacker_card, self.defender_card]

    @staticmethod
    def flatten_table_stack(table_stack: list[CardLoopStackItem]):
        tmp = []
        for stack_item in table_stack:
            attack_card, defender_card = stack_item.flatten()
            tmp.append(attack_card)
            if defender_card != void_card:
                tmp.append(defender_card)

        return tmp

class Player:
    trump_suit: Suit = Suit.VOID

    def __init__(self):
        # init start game
        self.cards: list[Card] = []

    def get_card(self, card) -> None:
        self.cards.append(card)

    def attack(self, table_stack: list[CardLoopStackItem]) -> Card:
        flatten_stack = CardLoopStackItem.flatten_table_stack(table_stack)
        action_cards_list = possible_attack_cards(flatten_stack, self.cards)
        attack_card = random.choice(action_cards_list)
        return attack_card

    def want_to_attack(self, table_stack):
        if len(table_stack) == 0:
            return True

    def __len__(self):
        return len(self.cards)

    def defend(self, table_stack: list[CardLoopStackItem]) -> Card:
        # True <-> player beat an attack card
        attack_card = table_stack[-1].attacker_card
        action_cards_list: list[Card] = card_methods.possible_defend_cards(attack_card, self.cards, Player.trump_suit)

        defend_card = random.choice(action_cards_list)
        if defend_card != void_card:
            self.cards.remove(defend_card)
        return defend_card


if __name__ == "__main__":
    game = DurakGame()
    game.add_player(Player())
    game.add_player(Player())
    game.add_player(Player())
