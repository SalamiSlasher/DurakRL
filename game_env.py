from __future__ import annotations

import random
from collections import deque

import torch
from gymnasium import Env

import card_methods
from card_methods import Card, Suit, void_card


DEBUG = True


def log_msg(msg, sep="=" * 50):
    if DEBUG:
        print(sep + str(msg) + sep)


class DurakEnv(Env): ...


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Bita(object):
    __metaclass__ = Singleton


class Take(object):
    __metaclass__ = Singleton


class DurakGame:
    def __init__(self) -> None:
        # self.player_count = player_count
        # self.players = deque(Player() for i in range(player_count))
        self.trump: Suit | None = None
        self.players: deque[Player] = deque(maxlen=6)
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
                to_take = self.deck.pop()
                player.get_card(to_take)

        self.trump: Suit = self.deck[0].suit
        Player.trump_suit = self.trump
        # --------------INIT--------------

        while not self.end_condition():
            log_msg("START ATTACK LOOP")
            # --------------TABLE_LOOP--------------

            attacker: Player = self.players[0]
            defender: Player = self.players[1]
            co_attacker: Player | None = (
                None if len(self.players) == 2 else self.players[-1]
            )

            for player in self.players:
                if DEBUG:
                    log_msg(player, "")

            turn_stack = []
            is_beaten_off: bool = self.attack_loop(
                attacker, defender, co_attacker, turn_stack, True, 0
            )
            # --------------TABLE_LOOP--------------

            # --------------GET_CARDS_FROM_DECK--------------
            if not is_beaten_off:
                to_take = CardLoopStackItem.flatten_table_stack(turn_stack)
                defender.get_cards(to_take)

            while len(self.deck) > 0 and any(
                len(player.cards) < 6 for player in self.players if player is not None
            ):
                to_take = self.deck.pop()
                if len(attacker.cards) != 6:
                    attacker.get_card(to_take)
                elif co_attacker and len(co_attacker.cards) != 6:
                    co_attacker.get_card(to_take)
                elif len(defender.cards) != 6:
                    defender.get_card(to_take)
                pass
            # --------------GET_CARDS_FROM_DECK--------------

            # --------------MAKE_ROTATION--------------
            self.players.append(self.players.popleft())
            # --------------MAKE_ROTATION--------------

            # --------------DEFINE_WINNERS--------------
            winners_tmp = []
            for player in self.players:
                if len(player.cards) == 0:
                    log_msg(f"PLAYER {player.name} wins!", "")
                    self.winners.append(player)
                    winners_tmp.append(player)

            for winner in winners_tmp:
                self.players.remove(winner)
            # --------------DEFINE_WINNERS--------------
            log_msg(
                f"END ATTACK LOOP with is_beaten: {is_beaten_off} and deck len: {len(self.deck)}"
            )
            pass

        print(f"GAME ENDS WITH WINNERS: {self.winners}\nand loser: {self.players}")

    def attack_loop(
        self,
        attacker,
        defender,
        co_attacker,
        turn_stack: list[CardLoopStackItem],
        is_beaten_off,
        turn,
    ) -> bool:
        if len(defender) - len(turn_stack) == 0:
            return is_beaten_off

        if not attacker.want_to_attack(turn_stack):
            if co_attacker and co_attacker.want_to_attack(turn_stack):
                return self.attack_loop(
                    co_attacker, defender, attacker, turn_stack, is_beaten_off, turn
                )
            return is_beaten_off

        turn = turn + 1
        # last card in turn_stack is card that must be beated by defender
        attack_card = attacker.attack(turn_stack)
        turn_stack_item = CardLoopStackItem(attack_card, attacker, defender)
        turn_stack.append(turn_stack_item)

        # Когда дефендер отбил
        if is_beaten_off:
            defend_card = defender.defend(turn_stack)
        # Когда дефендер не отбил в прошлом ходе и его топят...
        else:
            defend_card = void_card
        turn_stack[-1].defender_card = defend_card

        is_beaten_off = not card_methods.is_void_card(
            defend_card
        )  # void_card - не смог отбить
        log_msg(
            f"LOOP TURN #{turn}: |{attacker.name} ---> {defender.name}| {attack_card} ---> {defend_card}",
            sep="-" * 25,
        )
        log_msg(attacker, "")
        log_msg(co_attacker, "")
        log_msg(defender, "")
        log_msg(turn_stack, "")
        return self.attack_loop(
            attacker, defender, co_attacker, turn_stack, is_beaten_off, turn
        )

    def end_condition(self) -> bool:
        return len(self.players) <= 1

    def give_card(self, player: Player) -> None:
        player.get_card(self.deck.pop())


class CardLoopStackItem:
    def __init__(
        self,
        attacker_card: Card,
        attacker: Player,
        defender: Player,
    ):
        self.attacker_card = attacker_card
        self.attacker = attacker

        self.defender_card: Card = void_card
        self.defender = defender

    @property
    def id(self) -> int: ...

    def flatten(self):
        return [self.attacker_card, self.defender_card]

    def set_defend_card(self, defend_card: Card):
        self.defender_card = defend_card

    @staticmethod
    def flatten_table_stack(table_stack: list[CardLoopStackItem]):
        tmp = []
        for stack_item in table_stack:
            attack_card, defender_card = stack_item.flatten()
            tmp.append(attack_card)
            if not card_methods.is_void_card(defender_card):
                tmp.append(defender_card)

        return tmp

    def __str__(self):
        return f"({self.attacker_card}, {self.defender_card})"

    def __repr__(self):
        return self.__str__()


def is_possible_attack(card_in_hand: Card, desk: list[CardLoopStackItem]) -> bool:
    ranks_in_play = set()
    for card in desk:
        ranks_in_play.add(card.attacker_card.rank)
        if card.defender_card:
            ranks_in_play.add(card.defender_card.rank)

    return card_in_hand in ranks_in_play


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
        action_cards_list = card_methods.possible_attack_cards(
            flatten_stack, self.cards
        )
        attack_card = random.choice(action_cards_list)
        self.cards.remove(attack_card)
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
        action_cards_list: list[Card] = card_methods.possible_defend_cards(
            attack_card, self.cards, Player.trump_suit
        )

        defend_card = random.choice(action_cards_list)
        if not card_methods.is_void_card(defend_card):
            self.cards.remove(defend_card)
        return defend_card

    def __str__(self):
        return f"{self.name}: {self.cards}"

    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    DEBUG = True
    for i in range(1):
        random.seed(i)
        try:
            game = DurakGame()
            game.add_player(Player(name="ACTOR_1"))
            game.add_player(Player(name="ACTOR_2"))
            game.add_player(Player(name="ACTOR_3"))
            game.start_game()
        except IndexError:
            print(f"failed seed:= {i}")
            raise

        assert not game.players or len(game.players[0]) % 2 == 0, f"failed seed:= {i}"
        pass
