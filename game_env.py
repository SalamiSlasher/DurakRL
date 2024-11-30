from __future__ import annotations
import random

from gymnasium import Env, spaces
from collections import deque

from card_methods import Card, Suit, void_card
import card_methods

class DurakEnv(Env):
    ...


class DurakGame:
    def __init__(self):
        # self.player_count = player_count
        #self.players = deque(Player() for i in range(player_count))
        self.trump: Suit | None = None
        self.players = deque(maxlen=6)
        self.deck: list[Card] = card_methods.get_full_deck()
        self.beat: list[Card] = []

    def add_player(self, player: Player):
        self.players.append(player)

    def start_game(self):
        # --------------INIT--------------
        if len(self.players) < 2:
            raise AssertionError("Not enough players")

        for _ in range(6):
            for player in self.players:
                player.get_cards(self.deck)

        self.trump: Suit = self.deck[0].suit
        # --------------INIT--------------

        while self.end_condition():
            attacker = self.players[0]
            defender = self.players[1]
            co_attacker = None if len(self.players) == 2 else self.players[-1]

            turn_stack = []
            is_beaten_off = self.attack_loop(attacker, defender, co_attacker, turn_stack)

    def attack_loop(self, attacker, defender, co_attacker, turn_stack: list[CardLoopStackItem], is_beaten_off=True):
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

        turn_stack_item = CardLoopStackItem(attack_card, attacker, defend_card, defender)
        turn_stack.append(turn_stack_item)

        is_beaten_off = defend_card != void_card  # void_card - не смог отбить
        self.attack_loop(attacker, defender, co_attacker, turn_stack, is_beaten_off)

    def end_condition(self):
        return self.players == 1

    def give_card(self, player: Player):
        player.get_card(self.deck.pop())


class CardLoopStackItem:
    def __init__(self, attacker_card: Card, attacker: Player, defender_card: Card, defender: Player):
        self.attacker_card = attacker_card
        self.attacker = attacker

        self.defender_card: Card = defender_card
        self.defender = defender


class Player:
    def __init__(self):
        # init start game
        self.cards: list[Card] = []

    def get_card(self, card) -> None:
        self.cards.append(card)

    def attack(self, state) -> Card:
        return random.choice(self.cards)

    def want_to_attack(self, state):
        if len(state) == 0:
            return True

    def __len__(self):
        return len(self.cards)

    def defend(self, state) -> Card | None:
        # True <-> player beat an attack card
        to_defend = state[-1]
        return random.choice(self.cards)

if __name__ == '__main__':
    game = DurakGame()
    game.add_player(Player())
    game.add_player(Player())
    game.add_player(Player())
