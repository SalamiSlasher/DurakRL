from __future__ import annotations

import random
import numpy as np
from collections import deque
from typing import TypeAlias, NamedTuple, Any

import torch
from gymnasium import Env, spaces

import card_methods
from card_methods import Card, Suit, void_card, mapping

TABLE_TENSOR_LEN = 36 * 37
INPUT_LEN = (
    36  # hand
    + 36 * 37  # table
    + 36  # bita
    + 1  # is attacker
    + 4  # what suit is trump
)

OUTPUT_LEN = 38
DEBUG = True


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Bita(object):
    __metaclass__ = Singleton

    def __repr__(self) -> str:
        return 'Bita'


class Take(object):
    __metaclass__ = Singleton

    def __repr__(self) -> str:
        return 'Take'


Action: TypeAlias = Card | Bita | Take


def encode_action(action: Action, state: GameState) -> int:
    """
    Encodes an Action (Card, Bita, or Take) into an integer index.
    """
    if isinstance(action, Bita):
        return 0  # Index for Bita
    elif isinstance(action, Take):
        return 1  # Index for Take
    elif isinstance(action, Card):
        return 2 + state.hand.index(action)  # Cards are indexed starting from 2
    else:
        raise ValueError(f"Unknown action type: {action}")


def decode_action(action_index: int, state: GameState) -> Action:
    """
    Decodes an integer index into the corresponding Action (Card, Bita, or Take).
    """
    if action_index == 0:
        return Bita()
    elif action_index == 1:
        return Take()
    elif action_index >= 2 and action_index < 2 + len(state.hand):
        return state.hand[action_index - 2]
    else:
        raise ValueError(f"Invalid action index: {action_index}")



def log_msg(msg, sep="=" * 50):
    if DEBUG:
        print(sep + str(msg) + sep)


class GameState:
    def __init__(
        self,
        hand: list[Card],
        table: list[CardLoopStackItem],
        discard_pile: list[Card],
        seen_opponent_cards: list[Card],
        is_attacker: bool,
        trump: Suit,
    ) -> None:
        self.hand = hand
        self.table = table
        self.discard_pile = discard_pile
        self.seen_opponent_cards = seen_opponent_cards
        self.is_attacker = is_attacker
        self.trump = trump

    def to_tensor(self) -> torch.Tensor:
        hand_tensor = torch.zeros(36)  # Например, 36 карт в колоде
        for card in self.hand:
            hand_tensor[card.id] = 1

        table_tensor = torch.zeros(
            TABLE_TENSOR_LEN
        )  # Для каждой пары карты (атака + защита)
        for card_pair in self.table:
            idx = mapping[(card_pair.attacker_card, card_pair.defender_card)]
            table_tensor[idx] = 1

        discard_pile_tensor = torch.zeros(36)  # Битая колода
        for card in self.discard_pile:
            discard_pile_tensor[card.id] = 1

        # seen_opponent_cards_tensor = torch.zeros(
        #     36
        # )  # Карты противника, которые мы видели
        # for card in self.seen_opponent_cards:
        #     seen_opponent_cards_tensor[card.id] = 1

        is_attacker_tensor = torch.tensor([1.0 if self.is_attacker else 0.0])
        trump_tensor = torch.zeros(4)
        trump_tensor[int(self.trump)] = 1

        # Объединяем все тензоры в один вектор
        return torch.cat(
            [
                hand_tensor,
                table_tensor,
                discard_pile_tensor,
                # seen_opponent_cards_tensor,
                is_attacker_tensor,
                trump_tensor,
            ]
        )


class GameTensorState:
        def __init__(
            self,
            hand_tensor: torch.Tensor,
            table_tensor: torch.Tensor,
            discard_pile_tensor: torch.Tensor,
            is_attacker_tensor: torch.Tensor,
            trump_tensor: torch.Tensor,
        ):
            self.hand_tensor = hand_tensor
            self.table_tensor = table_tensor
            self.discard_pile_tensor = discard_pile_tensor
            self.is_attacker_tensor = is_attacker_tensor
            self.trump_tensor = trump_tensor

        def to_tensor(self) -> torch.Tensor:
            """
            Объединяет все тензоры в один вектор.
            """
            return torch.cat(
                [
                    self.hand_tensor,
                    self.table_tensor,
                    self.discard_pile_tensor,
                    self.is_attacker_tensor,
                    self.trump_tensor,
                ]
            )

        @staticmethod
        def from_game_state(state: GameState) -> GameTensorState:
            """
            Преобразует GameState в GameTensorState.
            """
            hand_tensor = torch.zeros(36)  # 36 карт в колоде
            for card in state.hand:
                hand_tensor[card.id] = 1

            table_tensor = torch.zeros(
                TABLE_TENSOR_LEN
            )  # Для пар карт (атака + защита)
            for card_pair in state.table:
                idx = mapping[(card_pair.attacker_card, card_pair.defender_card)]
                table_tensor[idx] = 1

            discard_pile_tensor = torch.zeros(36)  # Битые карты
            for card in state.discard_pile:
                discard_pile_tensor[card.id] = 1

            is_attacker_tensor = torch.tensor([1.0 if state.is_attacker else 0.0])
            trump_tensor = torch.zeros(4)  # Кодирование масти козыря
            trump_tensor[int(state.trump)] = 1

            return GameTensorState(
                hand_tensor,
                table_tensor,
                discard_pile_tensor,
                is_attacker_tensor,
                trump_tensor,
            )


class Transition(NamedTuple):
    state: GameState
    action: Action
    next_state: GameState
    reward: float
    done: bool


def create_attack_mask(hand_tensor: torch.Tensor, table_tensor: torch.Tensor) -> list[int]:
    """
    Создаёт маску действий для фазы атаки.
    """
    mask = [1] * OUTPUT_LEN
    mask[1] = 0  # Take запрещён для атакующего

    if table_tensor.sum().item() == 0:  # Если на столе ничего нет, можно играть любую карту
        return mask

    # Если карты уже на столе, атаковать можно только с совпадающим ранком
    valid_ranks = set()
    for idx, value in enumerate(table_tensor):
        if value == 1:
            attacker_card, defender_card = mapping[idx]
            valid_ranks.add(attacker_card.rank)

    for i, card_id in enumerate(hand_tensor.nonzero(as_tuple=True)[0]):
        card = Card.from_id(card_id.item())
        if card.rank not in valid_ranks:
            mask[i + 2] = 0  # Заблокировать карту

    return mask


def create_defense_mask(
    hand_tensor: torch.Tensor, table_tensor: torch.Tensor, trump_suit: int
) -> list[int]:
    """
    Создаёт маску действий для фазы защиты.
    """
    mask = [1] * OUTPUT_LEN
    mask[0] = 0  # Bita запрещено по умолчанию

    # Найти все атакующие карты, которые ещё нужно отбить
    attack_cards = []
    for idx, value in enumerate(table_tensor):
        if value == 1:
            attacker_card, defender_card = mapping[idx]
            if defender_card == void_card:  # Если карта не отбита
                attack_cards.append(attacker_card)

    if not attack_cards:  # Если нечего отбивать, разрешить Bita
        mask[0] = 1
        return mask

    # Проверить, чем можно отбить
    attack_card = attack_cards[0]  # Предполагаем, что на столе только одна карта для защиты
    for i, card_id in enumerate(hand_tensor.nonzero(as_tuple=True)[0]):
        card = Card.from_id(card_id.item())
        if not Card.can_beat(attack_card, card, Suit.from_int(trump_suit)):
            mask[i + 2] = 0  # Заблокировать карту

    return mask


def create_action_mask(
    hand_tensor: torch.Tensor,
    table_tensor: torch.Tensor,
    is_attacker: bool,
    trump_suit: int,
) -> list[int]:
    """
    Создаёт общую маску действий в зависимости от текущей фазы игры.
    """
    if is_attacker:
        return create_attack_mask(hand_tensor, table_tensor)
    else:
        return create_defense_mask(hand_tensor, table_tensor, trump_suit)


# def get_attack_action_mask(mask: list[int], state: GameState) -> list[int]:
#     mask[1] = 0  # cannot say Take
#
#     if len(state.table) == 0:  # no cards in play
#         return mask
#
#     for i, card in enumerate(state.hand):
#         if not is_possible_attack(card, state.table):
#             mask[i + 2] = 0
#
#     return mask
#
#
# def get_attack_action_mask_tensor(mask: list[int], state: GameTensorState) -> list[int]:
#     """
#     Обновляет маску для фазы атаки.
#     """
#     mask[1] = 0  # cannot say Take
#
#     if state.table_tensor.sum().item() == 0:  # No cards in play
#         return mask
#
#     # Проверяем карты в руке
#     for card_index in state.hand_tensor:
#         if not is_possible_attack(Card.from_id(card_index), state.table_tensor):
#             mask[i + 2] = 0
#
#     return mask
#
#
# def get_defender_action_mask_tensor(mask: list[int], state: GameTensorState) -> list[int]:
#     """
#     Обновляет маску для фазы защиты.
#     """
#     mask[0] = 0  # cannot say Bita
#
#     if state.table_tensor.sum().item() == 0:
#         raise AssertionError("Unreachable: cannot defend when there are no cards in play")
#
#     attack_cards = []
#     for idx, value in enumerate(state.table_tensor):
#         if value == 1:
#             attacker_card, defender_card = mapping[idx]
#             if defender_card == void_card:
#                 attack_cards.append(attacker_card)
#
#     if len(attack_cards) > 1:
#         raise AssertionError("Unreachable: more than one attack_card")
#
#     if len(attack_cards) < 1:
#         raise AssertionError("Unreachable: cannot defend when there are no cards in play")
#
#     for i, card in enumerate(state.hand_tensor.nonzero(as_tuple=True)[0]):
#         card_instance = Card.from_id(card.item())
#         if not Card.can_beat(
#             attack_card=attack_cards[0], defend_card=card_instance, trump=state.trump_tensor.argmax().item()
#         ):
#             mask[i + 2] = 0  # do not choose from these cards
#
#     return mask
#
#
# def get_action_mask_tensor(state: GameTensorState) -> list[int]:
#     """
#     Генерирует маску действий для текущего состояния игры.
#     """
#     mask = [1] * OUTPUT_LEN
#     # mask[0] == Bita  # first action in mark is always Bita
#     # mask[1] == Take  # second action in mark is always Take
#     # indices 2+ represent cards in hand
#
#     if state.is_attacker_tensor.item() == 1.0:
#         mask = get_attack_action_mask_tensor(mask, state)
#     else:
#         mask = get_attack_action_mask_tensor(mask, state)
#
#     return mask
#
#
#
# def get_defender_action_mask(mask: list[int], state: GameState) -> list[int]:
#     mask[0] = 0  # cannot say Bita
#
#     if not state.table:
#         raise AssertionError(
#             "Unreachable: cannot defend when there are no cards in play"
#         )
#
#     for i, card in enumerate(state.hand):
#         attack_cards = []
#         for card_pair in state.table:
#             if card_pair.defender_card == void_card:
#                 attack_cards.append(card_pair.attacker_card)
#
#         if len(attack_cards) > 1:
#             raise AssertionError("Unreachable: more than one attack_card")
#
#         if len(attack_cards) < 1:
#             raise AssertionError(
#                 "Unreachable: cannot defend when there are no cards in play"
#             )
#
#         if not Card.can_beat(
#             attack_card=attack_cards[0], defend_card=card, trump=state.trump
#         ):
#             mask[i + 2] = 0  # do not choose from these cards
#
#     return mask
#
#
# def get_action_mask(state: GameState) -> list[int]:
#     mask = [1] * (len(state.hand) + 2)
#     # mask[0] == Bita  # first action in mark is always Bita
#     # mask[1] == Take  # second action in mark is always Take
#     # isinstance(mask[2:], list[Card]) == True  # rest elements are Card objects
#
#     if state.is_attacker:
#         mask = get_attack_action_mask(mask, state)
#
#     else:
#         mask = get_defender_action_mask(mask, state)
#
#     return mask


class GamePhase:
    ATTACK = "attack"
    DEFENSE = "defense"
    END_TURN = "end_turn"
    ROTATE = "rotate"
    END_GAME = "end_game"


class DurakStateGame:
    def __init__(self) -> None:
        self.players = deque()
        self.deck = card_methods.get_full_deck()
        self.beat = []
        self.winners = []
        self.trump = None
        self.phase = GamePhase.ATTACK  # Start with the attack phase
        self.current_attacker = None
        self.current_defender = None
        self.turn_stack = []

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def initialize_game(self) -> None:
        if len(self.players) < 2:
            raise ValueError("Not enough players")

        # Deal cards
        for _ in range(6):
            for player in self.players:
                player.get_card(self.deck.pop())

        self.trump = self.deck[0].suit
        Player.trump_suit = self.trump

        # Set initial attacker and defender
        self.current_attacker = self.players[0]
        self.current_defender = self.players[1]

    def take_step(self, action: Action) -> None:
        """
        Execute a single step in the game based on the current phase and action.
        """
        if self.phase == GamePhase.ATTACK:
            self._process_attack(action)
        elif self.phase == GamePhase.DEFENSE:
            self._process_defense(action)
        elif self.phase == GamePhase.END_TURN:
            self._process_end_turn()
        elif self.phase == GamePhase.ROTATE:
            self._process_rotation()
        elif self.phase == GamePhase.END_GAME:
            raise RuntimeError("Game has ended. Reset to start a new game.")

    def _process_attack(self, action: Action) -> None:
        if isinstance(action, Card):
            # Attacker plays a card
            self.turn_stack.append(CardLoopStackItem(action, self.current_attacker, self.current_defender))
            self.current_attacker.cards.remove(action)
            self.phase = GamePhase.DEFENSE
        elif isinstance(action, Bita):
            # Defender chooses to defend off
            self.phase = GamePhase.END_TURN
        else:
            raise ValueError(f"Invalid action in attack phase: {action}")

    def _process_defense(self, action: Action):
        if isinstance(action, Card):
            # Defender plays a card
            attack_card = self.turn_stack[-1].attacker_card
            if not Card.can_beat(attack_card, action, self.trump):
                raise ValueError(f"Invalid defense: {action} cannot beat {attack_card}")

            self.turn_stack[-1].defender_card = action
            self.current_defender.cards.remove(action)
            self.phase = GamePhase.ATTACK
        elif isinstance(action, Take):
            # Defender chooses to take cards
            self.current_defender.get_cards(
                CardLoopStackItem.flatten_table_stack(self.turn_stack)
            )
            self.turn_stack.clear()
            self.phase = GamePhase.END_TURN
        elif isinstance(action, Bita):
            # Add a check to allow Bita only if all cards are defended
            if all(
                not card_methods.is_void_card(item.defender_card)
                for item in self.turn_stack
            ):
                self.phase = GamePhase.END_TURN
            else:
                raise ValueError(
                    "Bita is not allowed: some attack cards are not defended."
                )
        else:
            raise ValueError(f"Invalid action in defense phase: {action}")

    def _process_end_turn(self) -> None:
        # Clean up the table and prepare for the next turn
        if self.turn_stack:
            self.beat += CardLoopStackItem.flatten_table_stack(self.turn_stack)
            self.turn_stack.clear()

        if len(self.deck) > 0:
            for player in self.players:
                while len(player.cards) < 6 and self.deck:
                    player.get_card(self.deck.pop())

        self.phase = GamePhase.ROTATE

    def _process_rotation(self) -> None:
        # Rotate players and check for winners
        if len(self.current_defender.cards) == 0:
            self.winners.append(self.current_defender)
            self.players.remove(self.current_defender)

        if len(self.players) <= 1:
            self.phase = GamePhase.END_GAME
        else:
            self.players.rotate(-1)
            self.current_attacker = self.players[0]
            self.current_defender = self.players[1]
            self.phase = GamePhase.ATTACK

    def is_game_over(self) -> bool:
        return self.phase == GamePhase.END_GAME



class DurakEnv(Env):
    def __init__(self):
        super().__init__()
        self.game = DurakStateGame()
        self.game.add_player(Player("RL_AGENT"))
        self.game.add_player(Player("OPPONENT"))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(INPUT_LEN,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(OUTPUT_LEN)
        self.current_tensor_state = None
        self.action_mask = None  # Маска допустимых действий

    def reset(self):
        self.game.initialize_game()
        state = self._get_state(self.game.players[0])
        self.current_tensor_state = GameTensorState.from_game_state(state)
        self.action_mask = create_action_mask(
            hand_tensor=self.current_tensor_state.hand_tensor,
            table_tensor=self.current_tensor_state.table_tensor,
            trump_suit=int(state.trump),
            is_attacker=state.is_attacker,
        )
        return self.current_tensor_state.to_tensor().numpy(), {"action_mask": self.action_mask}

    def step(self, action: int):
        if self.action_mask[action] == 0:
            raise ValueError(f"Invalid action: {action}")

        if action == 0:  # Bita
            self.game.take_step(Bita())
        elif action == 1:  # Take
            self.game.take_step(Take())
        else:
            valid_hand_indices = self.current_tensor_state.hand_tensor.nonzero(
                as_tuple=True
            )[0]
            hand_index = action - 2

            if hand_index >= len(valid_hand_indices):
                raise IndexError(
                    f"Invalid hand index: {hand_index}, valid indices: {valid_hand_indices.tolist()}"
                )

            card_id = valid_hand_indices[hand_index].item()
            self.game.take_step(Card.from_id(card_id))

        # Обновляем состояние
        state = self._get_state(self.game.players[0])
        self.current_tensor_state = GameTensorState.from_game_state(state)

        # Создаём новую маску действий
        self.action_mask = create_action_mask(
            self.current_tensor_state.hand_tensor,
            self.current_tensor_state.table_tensor,
            self.current_tensor_state.is_attacker_tensor.item() == 1.0,
            self.current_tensor_state.trump_tensor.argmax().item(),
        )

        reward = 1.0 if self.game.phase == GamePhase.END_TURN else 0.0
        done = self.game.is_game_over()

        return (
            self.current_tensor_state.to_tensor().numpy(),
            reward,
            done,
            False,
            {"action_mask": self.action_mask},
        )

    def _get_state(self, player: Player):
        """
        Генерирует текущее состояние игры для заданного игрока.
        """
        return GameState(
            hand=player.cards,
            table=self.game.turn_stack,
            discard_pile=self.game.beat,
            seen_opponent_cards=[],  # Добавить логику для видимых карт противника
            is_attacker=self.game.players[0] == player,
            trump=self.game.trump,
        )

    def _get_action_mask(self):
        """
        Генерирует маску действий на основе текущей фазы игры.
        """
        if self.game.phase == GamePhase.ATTACK:
            return get_attack_action_mask([1] * OUTPUT_LEN, self.current_tensor_state)
        elif self.game.phase == GamePhase.DEFENSE:
            return get_defender_action_mask([1] * OUTPUT_LEN, self.current_tensor_state)
        else:
            return [0] * OUTPUT_LEN  # Нет доступных действий в других фазах

    def _decode_action(self, action_index: int) -> Action:
        """
        Декодирует индекс действия в объект действия (Bita, Take, Card).
        """
        if action_index == 0:
            return Bita()
        elif action_index == 1:
            return Take()
        elif action_index >= 2 and action_index < 2 + len(self.current_tensor_state.hand_tensor.nonzero()):
            return self.current_tensor_state.hand_tensor.nonzero()[action_index - 2].item()
        else:
            raise ValueError(f"Invalid action index: {action_index}")



def encode_action_tensor(action: Action, state: GameState) -> torch.Tensor:
    """
    Кодирует действие (Bita, Take, Card) в виде тензора.
    """
    action_tensor = torch.zeros(OUTPUT_LEN)
    if isinstance(action, Bita):
        action_tensor[0] = 1
    elif isinstance(action, Take):
        action_tensor[1] = 1
    elif isinstance(action, Card):
        card_index = 2 + state.hand.index(action)
        action_tensor[card_index] = 1
    else:
        raise ValueError(f"Unknown action type: {action}")
    return action_tensor



def decode_action_tensor(action_tensor: torch.Tensor, state: GameState) -> Action:
    """
    Декодирует тензор действия в объект действия (Bita, Take, Card).
    """
    action_index = torch.argmax(action_tensor).item()
    if action_index == 0:
        return Bita()
    elif action_index == 1:
        return Take()
    elif action_index >= 2 and action_index < 2 + len(state.hand):
        return state.hand[action_index - 2]
    else:
        raise ValueError(f"Invalid action index in tensor: {action_index}")




# class DurakGame:
#     def __init__(self) -> None:
#         # self.player_count = player_count
#         # self.players = deque(Player() for i in range(player_count))
#         self.trump: Suit | None = None
#         self.players: deque[Player] = deque(maxlen=6)
#         self.deck: list[Card] = card_methods.get_full_deck()
#         self.beat: list[Card] = []
#
#         self.winners: list[Player] = []
#
#     def add_player(self, player: Player) -> None:
#         self.players.append(player)
#
#     def start_game(self) -> None:
#         # --------------INIT--------------
#         if len(self.players) < 2:
#             raise AssertionError("Not enough players")
#
#         for _ in range(6):
#             for player in self.players:
#                 to_take = self.deck.pop()
#                 player.get_card(to_take)
#
#         self.trump: Suit = self.deck[0].suit
#         Player.trump_suit = self.trump
#         # --------------INIT--------------
#
#         while not self.end_condition():
#             log_msg("START ATTACK LOOP")
#             # --------------TABLE_LOOP--------------
#
#             attacker: Player = self.players[0]
#             defender: Player = self.players[1]
#             co_attacker: Player | None = (
#                 None if len(self.players) == 2 else self.players[-1]
#             )
#
#             for player in self.players:
#                 if DEBUG:
#                     log_msg(player, "")
#
#             turn_stack = []
#             is_beaten_off: bool = self.attack_loop(
#                 attacker, defender, co_attacker, turn_stack, True, 0
#             )
#             # --------------TABLE_LOOP--------------
#
#             # --------------GET_CARDS_FROM_DECK--------------
#             if not is_beaten_off:
#                 to_take = CardLoopStackItem.flatten_table_stack(turn_stack)
#                 defender.get_cards(to_take)
#
#             while len(self.deck) > 0 and any(
#                 len(player.cards) < 6 for player in self.players if player is not None
#             ):
#                 to_take = self.deck.pop()
#                 if len(attacker.cards) != 6:
#                     attacker.get_card(to_take)
#                 elif co_attacker and len(co_attacker.cards) != 6:
#                     co_attacker.get_card(to_take)
#                 elif len(defender.cards) != 6:
#                     defender.get_card(to_take)
#                 pass
#             # --------------GET_CARDS_FROM_DECK--------------
#
#             # --------------MAKE_ROTATION--------------
#             self.players.append(self.players.popleft())
#             # --------------MAKE_ROTATION--------------
#
#             # --------------DEFINE_WINNERS--------------
#             winners_tmp = []
#             for player in self.players:
#                 if len(player.cards) == 0:
#                     log_msg(f"PLAYER {player.name} wins!", "")
#                     self.winners.append(player)
#                     winners_tmp.append(player)
#
#             for winner in winners_tmp:
#                 self.players.remove(winner)
#             # --------------DEFINE_WINNERS--------------
#             log_msg(
#                 f"END ATTACK LOOP with is_beaten: {is_beaten_off} and deck len: {len(self.deck)}"
#             )
#             pass
#
#         print(f"GAME ENDS WITH WINNERS: {self.winners}\nand loser: {self.players}")
#
#     def attack_loop(
#         self,
#         attacker,
#         defender,
#         co_attacker,
#         turn_stack: list[CardLoopStackItem],
#         is_beaten_off,
#         turn,
#     ) -> bool:
#         if len(defender) - len(turn_stack) == 0:
#             return is_beaten_off
#
#         if not attacker.want_to_attack(turn_stack):
#             if co_attacker and co_attacker.want_to_attack(turn_stack):
#                 return self.attack_loop(
#                     co_attacker, defender, attacker, turn_stack, is_beaten_off, turn
#                 )
#             return is_beaten_off
#
#         turn = turn + 1
#         # last card in turn_stack is card that must be beated by defender
#         attack_card = attacker.attack(turn_stack)
#         turn_stack_item = CardLoopStackItem(attack_card, attacker, defender)
#         turn_stack.append(turn_stack_item)
#
#         # Когда дефендер отбил
#         if is_beaten_off:
#             defend_card = defender.defend(turn_stack)
#         # Когда дефендер не отбил в прошлом ходе и его топят...
#         else:
#             defend_card = void_card
#         turn_stack[-1].defender_card = defend_card
#
#         is_beaten_off = not card_methods.is_void_card(
#             defend_card
#         )  # void_card - не смог отбить
#         log_msg(
#             f"LOOP TURN #{turn}: |{attacker.name} ---> {defender.name}| {attack_card} ---> {defend_card}",
#             sep="-" * 25,
#         )
#         log_msg(attacker, "")
#         log_msg(co_attacker, "")
#         log_msg(defender, "")
#         log_msg(turn_stack, "")
#         return self.attack_loop(
#             attacker, defender, co_attacker, turn_stack, is_beaten_off, turn
#         )
#
#     def end_condition(self) -> bool:
#         return len(self.players) <= 1
#
#     def give_card(self, player: Player) -> None:
#         player.get_card(self.deck.pop())


class CardLoopStackItem:
    def __init__(
        self,
        attacker_card: Card,
        # attacker: Player,
        # defender: Player,
    ):
        self.attacker_card = attacker_card
        # self.attacker = attacker

        self.defender_card: Card = void_card
        # self.defender = defender

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
    DEBUG = False
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
