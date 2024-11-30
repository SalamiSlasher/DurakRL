from __future__ import annotations
import random

from gymnasium import Env
from collections import deque

from pycparser.c_ast import While

from card_methods import Card, Suit, void_card, possible_attack_cards
import card_methods


def log_msg(msg, sep='=' * 50):
    print(sep + msg + sep)

class DurakEnv(Env): ...


class DurakGame:
    def __init__(self) -> None:
        # self.player_count = player_count
        # self.players = deque(Player() for i in range(player_count))
        self.trump: Suit | None = None
        self.players = deque(maxlen=6)
        self.deck: list[Card] = card_methods.get_full_deck()
        self.beat: list[Card] = []

        self.winners = []

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

        while not self.end_condition():
            log_msg('START ATTACK LOOP')
            # --------------TABLE_LOOP--------------

            attacker: Player = self.players[0]
            defender: Player = self.players[1]
            co_attacker: Player | None = None if len(self.players) == 2 else self.players[-1]

            for player in self.players:
                print(player)

            turn_stack = []
            is_beaten_off: bool = self.attack_loop(
                attacker, defender, co_attacker, turn_stack
            )
            # --------------TABLE_LOOP--------------

            # --------------GET_CARDS_FROM_DECK--------------
            if not is_beaten_off:
                to_take = CardLoopStackItem.flatten_table_stack(turn_stack)
                defender.get_cards(to_take)

            while len(self.deck) > 0 and any(len(player.cards) != 6 for player in self.players if player is not None):
                to_take = self.deck.pop()
                if len(attacker.cards) != 6:
                    attacker.get_card(to_take)
                elif co_attacker and len(co_attacker.cards) != 6:
                    co_attacker.get_card(to_take)
                elif len(defender.cards) != 6:
                    defender.get_card(to_take)
            # --------------GET_CARDS_FROM_DECK--------------

            # --------------MAKE_ROTATION--------------
            self.players.append(self.players.popleft())
            # --------------MAKE_ROTATION--------------

            # --------------DEFINE_WINNERS--------------
            for player in self.players:
                if len(player.cards) == 0:
                    print(f'PLAYER {player.name} wins!')
                    self.winners.append(player)
                    self.players.remove(player)
            # --------------DEFINE_WINNERS--------------
            log_msg('END ATTACK LOOP')

    def attack_loop(
        self,
        attacker,
        defender,
        co_attacker,
        turn_stack: list[CardLoopStackItem],
        is_beaten_off=True,
    ) -> bool:
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
        log_msg(f'{attacker.name} ---> {defender.name}: {attack_card} ---> {defend_card}', sep='-' * 25)
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

    def __init__(self, name: str):
        # init start game
        self.name = name
        self.cards: list[Card] = []

    def get_card(self, card) -> None:
        self.cards.append(card)

    def get_cards(self, cards: list[Card]) -> None:
        for card in cards:
            self.get_card(card)

    def attack(self, table_stack: list[CardLoopStackItem]) -> Card:
        flatten_stack = CardLoopStackItem.flatten_table_stack(table_stack)
        action_cards_list = card_methods.possible_attack_cards(flatten_stack, self.cards)
        attack_card = random.choice(action_cards_list)
        return attack_card

    def want_to_attack(self, table_stack):
        if len(table_stack) == 0:
            return True

        flatten_stack = CardLoopStackItem.flatten_table_stack(table_stack)
        return len(card_methods.possible_attack_cards(flatten_stack, self.cards)) > 0

    def __len__(self):
        return len(self.cards)

    def defend(self, table_stack: list[CardLoopStackItem]) -> Card:
        attack_card = table_stack[-1].attacker_card
        action_cards_list: list[Card] = card_methods.possible_defend_cards(attack_card, self.cards, Player.trump_suit)

        defend_card = random.choice(action_cards_list)
        if defend_card != void_card:
            self.cards.remove(defend_card)
        return defend_card

    def __str__(self):
        return f'{self.name}: {self.cards}'


if __name__ == "__main__":
    game = DurakGame()
    game.add_player(Player(name='ACTOR_1'))
    game.add_player(Player(name='ACTOR_2'))
    game.add_player(Player(name='ACTOR_3'))
    game.start_game()
