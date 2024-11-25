from __future__ import annotations
import random

from gymnasium import Env, spaces
from collections import deque

from card_methods import Card
import card_methods

class DurakEnv(Env):
    ...


class DurakGame:
    def __init__(self):
        # self.player_count = player_count
        #self.players = deque(Player() for i in range(player_count))
        self.trump = None
        self.players = deque(maxlen=6)
        self.deck: list[Card] = card_methods.get_full_deck()

    def add_player(self, player: Player):
        self.players.append(player)

    def start_game(self):
        # --------------INIT--------------
        if len(self.players) < 2:
            raise AssertionError("Not enough players")

        for _ in range(6):
            for player in self.players:
                player.get_cards(self.deck)

        self.trump = self.deck[0]
        # --------------INIT--------------

        while self.end_condition():
            attacker = self.players[0]
            defender = self.players[1]
            co_attacker = None if len(self.players) == 2 else self.players[-1]

            self.attack_loop(attacker, defender, co_attacker)

    def attack_loop(self, attacker, defender, co_attacker, turn_stack=[], is_beaten_off=False):
        if not attacker.want_to_attack(turn_stack):
            if co_attacker and co_attacker.want_to_attack(turn_stack):
                return self.attack_loop(co_attacker, defender, attacker, turn_stack)
            return

        # last card in turn_stack is card that must be beated by defender
        attacker.attack(turn_stack)
        is_beaten_off = defender.defend(turn_stack) # отбил ли? True <-> да

        self.attack_loop(attacker, defender, co_attacker, turn_stack, is_beaten_off)

            if is_beaten_off:
                self.attack_loop(attacker, defender, co_attacker, turn_stack, is_beaten_off)
            else:
                pass





    def end_condition(self):
        return self.players == 1

    def give_card(self, player: Player):
        player.get_card(self.deck.pop())


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

    def defend(self, state) -> bool:
        # True <-> player beat an attack card
        to_defend = state[-1]
        return False

if __name__ == '__main__':
    game = DurakGame()
    game.add_player(Player())
    game.add_player(Player())
    game.add_player(Player())
