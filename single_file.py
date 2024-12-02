from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import Final
from typing import NamedTuple
from typing import Protocol
from typing import TypeAlias

from collections import deque
from enum import Enum
from enum import IntEnum
from random import choice
from random import shuffle

import itertools
import math
import random

from gymnasium import Env
from gymnasium import spaces
from torch import nn
from torch import optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch


def log_msg(msg: object, sep: str = '=' * 50) -> None:
    if DEBUG:
        print(sep + str(msg) + sep)


device: torch.device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

TABLE_TENSOR_LEN: Final[int] = 36 * 37
INPUT_LEN: Final[int] = (
    36  # hand
    + 36 * 37  # table
    + 36  # bita
    + 1  # is attacker
    + 4  # what suit is trump
)

OUTPUT_LEN: Final[int] = 38
DEBUG = True
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
steps_done = 0

TARGET_UPDATE = 100


class Singleton(type):
    _instances: ClassVar[dict[str, object]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> object:
        if cls not in cls._instances:  # type: ignore[comparison-overlap]
            cls._instances[cls] = super(Singleton, cls).__call__(  # type: ignore[index]
                *args, **kwargs
            )
        return cls._instances[cls]  # type: ignore[index]


class Bita:
    __metaclass__ = Singleton

    def __repr__(self) -> str:
        return 'Bita'


class Take:
    __metaclass__ = Singleton

    def __repr__(self) -> str:
        return 'Take'


class Suit(Enum):
    HEARTS = '♡'
    DIAMONDS = '♢'
    CLUBS = '♣'
    SPADES = '♠'
    VOID = 'void'

    def __int__(self) -> int:
        match self:
            case Suit.VOID:
                return -1
            case Suit.HEARTS:
                return 0
            case Suit.DIAMONDS:
                return 1
            case Suit.CLUBS:
                return 2
            case Suit.SPADES:
                return 3
            case _:
                raise ValueError('Invalid Suit int.')  # noqa: TRY003

    def __float__(self) -> float:
        return float(int(self))

    @classmethod
    def from_int(cls, suit_integer: int) -> Suit:
        match suit_integer:
            case -1:
                return Suit.VOID
            case 0:
                return Suit.HEARTS
            case 1:
                return Suit.DIAMONDS
            case 2:
                return Suit.CLUBS
            case 3:
                return Suit.SPADES
            case _:
                raise ValueError('Invalid Suit int.')  # noqa: TRY003


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


VOID_CARD_ID: Final[int] = 36


class Card:
    encoder: Final[dict[Rank, str]] = {
        Rank.SIX: '6',
        Rank.SEVEN: '7',
        Rank.EIGHT: '8',
        Rank.NINE: '9',
        Rank.TEN: '10',
        Rank.JACK: 'J',
        Rank.QUEEN: 'Q',
        Rank.KING: 'K',
        Rank.ACE: 'A',
        Rank.VOID: 'VOID',
    }

    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    def __repr__(self) -> str:
        return f'{self.encoder[self.rank]}{self.suit.value}'

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

    @property
    def id(self) -> int:
        """
        Возвращает уникальный идентификатор карты.
        Идентификатор - это целое число, которое зависит от масти и достоинства.
        """
        if is_void_card(self):
            return VOID_CARD_ID
        return (self.rank.value - 6) + (int(self.suit) * 9)

    @classmethod
    def from_id(cls, card_id: int) -> Card:
        if card_id == VOID_CARD_ID:
            return VOID_CARD

        suit_int = card_id // 9  # Integer division to get the suit
        rank_value = (
            card_id % 9
        ) + 6  # Get the rank by using the remainder and adding 6

        # Assuming `Rank` and `Suit` are classes or enums:
        rank = Rank(rank_value)
        suit = Suit.from_int(suit_int)

        return cls(rank=rank, suit=suit)

    @staticmethod
    def can_beat(attack_card: Card, defend_card: Card, trump: Suit) -> bool:
        if attack_card.suit == defend_card.suit:
            return defend_card.rank > attack_card.rank
        return defend_card.suit == trump and attack_card.suit != trump

    def can_this_beat(self, attack_card: Card, trump: Suit) -> bool:
        return Card.can_beat(attack_card, self, trump)


VOID_CARD: Final[Card] = Card(Suit.VOID, Rank.VOID)


def is_void_card(card: Card) -> bool:
    return card.rank == Rank.VOID or card.suit == Suit.VOID


def get_full_deck() -> list[Card]:
    deck = [
        Card(suit, rank)
        for suit, rank in itertools.product(Suit, Rank)
        if not is_void_card(Card(suit, rank))
    ]
    shuffle(deck)
    return deck


def generate_full_table_states() -> list[tuple[Card, Card]]:
    return list(
        itertools.product(get_full_deck(), [*get_full_deck(), VOID_CARD])
    )


def possible_attack_cards(
    card_stack: list[Card], player_cards: list[Card]
) -> list[Card]:
    if len(card_stack) == 0:
        return player_cards

    # Собираем ранги всех карт на столе, исключая пустые карты
    table_ranks = {card.rank for card in card_stack if not is_void_card(card)}
    # Возвращаем карты из руки игрока, ранги которых есть на столе
    return [card for card in player_cards if card.rank in table_ranks]


def possible_defend_cards(
    attack_card: Card, player_cards: list[Card], trump: Suit
) -> list[Card]:
    # Фильтруем карты игрока, которые могут побить атакующую карту
    defend_cards = [
        card for card in player_cards if Card.can_beat(attack_card, card, trump)
    ]
    defend_cards.append(VOID_CARD)
    return defend_cards


full_table_states = generate_full_table_states()
table_state_to_int_mapping: dict[tuple[Card, Card], int] = {}
for i in range(len(full_table_states)):
    table_state_to_int_mapping[full_table_states[i]] = i

# ===========================


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Actor(Protocol):
    def __call__(self, state: torch.Tensor) -> torch.Tensor: ...

    policy_net: DQN


class DQN(nn.Module):
    def __init__(
        self, n_observations: int = INPUT_LEN, n_actions: int = OUTPUT_LEN
    ):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))

        logits = self.layer3(x)

        return F.softmax(logits, dim=-1)


class GameTensorAction:
    def __init__(self, action_index: int):
        """
        Инициализирует действие.
        :param action_index: Индекс действия (0 для Бита, 1 для Взять, 2-37 для карт).
        """
        assert 0 <= action_index < 38, (
            'action_index должен быть в диапазоне 0-37'
        )
        self.action_index = action_index

    def to_tensor(self) -> torch.Tensor:
        """
        Преобразует действие в one-hot тензор длины 38.
        """
        action_tensor = torch.zeros(
            38
        )  # Размер 38: 0 (Бита), 1 (Взять), 2-37 (карты)
        action_tensor[self.action_index] = 1.0
        return action_tensor

    @classmethod
    def from_tensor(cls, action_tensor: torch.Tensor) -> GameTensorAction:
        """
        Преобразует тензор обратно в действие.
        :param action_tensor: Тензор длины 38.
        """
        assert action_tensor.size() == (38,), (
            'action_tensor должен быть длины 38'
        )
        action_index: int = torch.argmax(action_tensor).item()
        return cls(action_index)

    def __repr__(self) -> str:
        """
        Представление действия в удобочитаемом формате.
        """
        if self.action_index == 0:
            return 'Action: Бита'
        if self.action_index == 1:
            return 'Action: Взять'
        else:
            return f'Action: Карта {self.action_index - 2}'  # Смещение для карт


def generate_all_possible_actions() -> torch.Tensor:
    """
    Генерирует все возможные действия в виде тензоров.
    """
    return torch.stack([GameTensorAction(idx).to_tensor() for idx in range(39)])


def create_attack_mask(
    hand_tensor: torch.Tensor, table_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Создаёт маску действий для фазы атаки.
    :param hand_tensor: Тензор руки длиной 36.
    :param table_tensor: Тензор стола размером 36x37.
    :return: Маска действий длиной 38.
    """
    mask = torch.ones(38, dtype=torch.float32)
    mask[1] = 0.0  # Атакующему нельзя брать карты

    # Если на столе ничего нет, можно играть любую карту
    if table_tensor.sum().item() == 0:
        mask[2:] = hand_tensor
        return mask

    # Иначе атаковать можно только картами с совпадающим ранком
    table_ranks = set()
    for idx in range(36):
        if table_tensor[idx].sum().item() > 0:  # Есть карта на столе
            card = Card.from_id(idx)
            table_ranks.add(card.rank)

    for idx in range(36):
        card = Card.from_id(idx)
        if hand_tensor[idx] > 0 and card.rank in table_ranks:
            mask[idx + 2] = 1.0
        else:
            mask[idx + 2] = 0.0

    return mask


def create_defense_mask(
    hand_tensor: torch.Tensor, table_tensor: torch.Tensor, trump_suit: Suit
) -> torch.Tensor:
    """
    Создаёт маску действий для фазы защиты.
    :param hand_tensor: Тензор руки длиной 36.
    :param table_tensor: Тензор стола размером 36x37.
    :param trump_suit: Козырная масть.
    :return: Маска действий длиной 38.
    """
    mask = torch.ones(38, dtype=torch.float32)
    mask[0] = 0.0  # Защищающемуся нельзя сбросить карты по умолчанию

    # Найти все атакующие карты, которые ещё нужно отбить
    attack_cards = []
    for idx in range(36):
        if table_tensor[idx].sum().item() > 0:  # Есть карта на столе
            for defender_idx in range(37):
                if table_tensor[idx, defender_idx] == 0:  # Карта не отбита
                    attack_cards.append(Card.from_id(idx))
                    break

    if not attack_cards:  # Если нечего отбивать, разрешить Bita
        mask[0] = 1.0
        return mask

    # Проверить, чем можно отбить первую атакующую карту
    attack_card = attack_cards[
        0
    ]  # Предполагаем, что защищаемся от первой карты
    for idx in range(36):
        card = Card.from_id(idx)
        if hand_tensor[idx] > 0 and Card.can_beat(
            attack_card, card, trump_suit
        ):
            mask[idx + 2] = 1.0
        else:
            mask[idx + 2] = 0.0

    return mask


class GameTensorState:
    def __init__(
        self,
        hand_tensor: torch.Tensor,  # Тензор длины 36
        table_tensor: torch.Tensor,  # Тензор размера 36 * 37
        discard_pile_tensor: torch.Tensor,  # Тензор длины 36
        is_attacker_tensor: torch.Tensor,  # Тензор длины 1 (булевый индикатор)
        trump_tensor: torch.Tensor,  # Тензор длины 4 (масть козыря)
    ):
        assert hand_tensor.size() == (36,), 'hand_tensor должен быть длины 36'
        assert table_tensor.size() == (36, 37), (
            'table_tensor должен быть размером 36x37'
        )
        assert discard_pile_tensor.size() == (36,), (
            'discard_pile_tensor должен быть длины 36'
        )
        assert is_attacker_tensor.size() == (1,), (
            'is_attacker_tensor должен быть длины 1'
        )
        assert trump_tensor.size() == (4,), 'trump_tensor должен быть длины 4'

        self.hand_tensor = hand_tensor
        self.table_tensor = table_tensor
        self.discard_pile_tensor = discard_pile_tensor
        self.is_attacker_tensor = is_attacker_tensor
        self.trump_tensor = trump_tensor

    def to_tensor(self) -> torch.Tensor:
        """
        Объединяет все тензоры в один вектор для использования в модели.
        """
        return torch.cat([
            self.hand_tensor,
            self.table_tensor.flatten(),  # Превращаем стол в одномерный тензор
            self.discard_pile_tensor,
            self.is_attacker_tensor,
            self.trump_tensor,
        ])

    def update_hand(self, card_index: int, *, give: bool) -> None:
        """
        Обновляет состояние руки, добавляя или убирая карту.
        :param card_index: Индекс карты (0-35).
        :param give: True, если карта добавляется; False, если убирается.
        """
        self.hand_tensor[card_index] = 1.0 if give else 0.0

    def update_table(self, attack_card: int, defend_card: int) -> None:
        """
        Обновляет состояние стола.
        :param attack_card: Индекс карты атаки (0-35).
        :param defend_card: Индекс карты защиты (0-36, где 36 — VOID_CARD).
        """
        self.table_tensor[attack_card, defend_card] = 1.0

    def update_discard_pile(self, card_index: int, *, discard: bool) -> None:
        """
        Обновляет состояние сброса.
        :param card_index: Индекс карты (0-35).
        :param discard: True, если карта добавлена в сброс; False, если удалена.
        """
        self.discard_pile_tensor[card_index] = 1.0 if discard else 0.0

    def set_trump(self, trump_suit: int) -> None:
        """
        Устанавливает козырь.
        :param trump_suit: Индекс масти козыря (0-3).
        """
        self.trump_tensor.zero_()
        self.trump_tensor[trump_suit] = 1.0

    def set_is_attacker(self, is_attacker: bool) -> None:
        """
        Устанавливает, является ли текущий игрок атакующим.
        :param is_attacker: True, если игрок атакующий, иначе False.
        """
        self.is_attacker_tensor[0] = 1.0 if is_attacker else 0.0


class Transition(NamedTuple):
    state: GameTensorState
    action: GameTensorAction
    next_state: GameTensorState
    reward: float
    done: bool


class GamePhase:
    ATTACK = 'attack'
    DEFENSE = 'defense'
    END_TURN = 'end_turn'
    ROTATE = 'rotate'
    END_GAME = 'end_game'


class Player(Protocol):
    hand_tensor: torch.Tensor
    trump_suit: torch.Tensor

    def __init__(self, name: str) -> None: ...

    def act(self, state: GameTensorState) -> GameTensorAction: ...

    def get_card_count(self) -> int: ...

    def take_card(self, card_index: int) -> None: ...

    def take_cards(self, table_tensor: torch.Tensor) -> None: ...

    @property
    def cards(self) -> torch.Tensor: ...


class DurakTensorGame:
    def __init__(self) -> None:
        self.players: deque[Player] = deque()
        self.deck = torch.randperm(36)  # Индексы карт в перетасованной колоде
        self.discard_pile = torch.zeros(36, dtype=torch.float32)
        self.turn_stack = torch.zeros(
            36, 37, dtype=torch.float32
        )  # Стол: карты атаки/защиты
        self.trump_suit = None  # Козырная масть
        self.phase = GamePhase.ATTACK  # Начальная фаза
        self.current_attacker: Player | None = None
        self.current_defender: Player | None = None

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def initialize_game(self) -> None:
        if len(self.players) < 2:  # noqa: PLR2004
            msg = 'Not enough players'
            raise ValueError(msg)

        # Раздаём карты игрокам
        for _ in range(6):
            for player in self.players:
                card_index = self.deck[-1].item()
                player.take_card(card_index)
                self.deck = self.deck[:-1]

        # Устанавливаем козырь
        trump_card_index: int = self.deck[0].item()
        self.trump_suit = trump_card_index // 9
        Player.trump_suit = self.trump_suit

        # Назначаем атакующего и защищающегося
        self.current_attacker = self.players[0]
        self.current_defender = self.players[1]

    def take_step(self, action_tensor: torch.Tensor) -> None:
        """
        Выполняет шаг игры на основе текущей фазы и действия.
        """
        action_index = torch.argmax(action_tensor).item()
        if self.phase == GamePhase.ATTACK:
            self._process_attack(action_index)
        elif self.phase == GamePhase.DEFENSE:
            self._process_defense(action_index)
        elif self.phase == GamePhase.END_TURN:
            self._process_end_turn()
        elif self.phase == GamePhase.ROTATE:
            self._process_rotation()
        elif self.phase == GamePhase.END_GAME:
            msg = 'Game has ended. Reset to start a new game.'
            raise RuntimeError(msg)

    def _process_attack(self, action_index: int) -> None:
        if action_index >= 2:  # Карта
            attack_card_index = action_index - 2
            self.turn_stack[attack_card_index, :] = (
                1.0  # Добавляем карту на стол
            )
            self.phase = GamePhase.DEFENSE
        else:
            msg = f'Invalid action in attack phase: {action_index}'
            raise ValueError(msg)

    def _process_defense(self, action_index: int) -> None:
        if action_index >= 2:  # Карта  # noqa: PLR2004
            defend_card_index = action_index - 2
            attack_card_index = self._get_last_attack_card_index()
            if not self._can_beat(attack_card_index, defend_card_index):
                msg = f'Invalid defense: card {defend_card_index} cannot beat {attack_card_index}'
                raise ValueError(msg)
            self.turn_stack[attack_card_index, defend_card_index] = (
                1.0  # Обновляем стол
            )
            self.phase = GamePhase.ATTACK
        elif action_index == 1:  # Взять карты
            self.current_defender.take_cards(self._get_table_cards())
            self._clear_turn_stack()
            self.phase = GamePhase.END_TURN
        else:
            msg = f'Invalid action in defense phase: {action_index}'
            raise ValueError(msg)

    def _process_end_turn(self) -> None:
        # Перемещаем карты со стола в сброс
        self.discard_pile += self._get_table_cards()
        self._clear_turn_stack()

        # Добираем карты из колоды
        if len(self.deck) > 0:
            for player in self.players:
                while player.get_card_count() < 6 and len(self.deck) > 0:  # noqa: PLR2004
                    card_index = self.deck[-1].item()
                    player.take_card(card_index)
                    self.deck = self.deck[:-1]

        self.phase = GamePhase.ROTATE

    def _process_rotation(self) -> None:
        if len(self.current_defender.cards) == 0:
            self.players.remove(self.current_defender)
            if len(self.players) <= 1:
                self.phase = GamePhase.END_GAME
                return

        self.players.rotate(-1)
        self.current_attacker = self.players[0]
        self.current_defender = self.players[1]
        self.phase = GamePhase.ATTACK

    def _can_beat(self, attack_card_index: int, defend_card_index: int) -> bool:
        attack_card = Card.from_id(attack_card_index)
        defend_card = Card.from_id(defend_card_index)
        return Card.can_beat(
            attack_card, defend_card, Suit.from_int(self.trump_suit)
        )

    def _get_table_cards(self) -> torch.Tensor:
        return self.turn_stack.sum(dim=1)

    def _clear_turn_stack(self) -> None:
        self.turn_stack.zero_()

    def _get_last_attack_card_index(self) -> int:
        for idx in range(36):
            if (
                self.turn_stack[idx, :].sum().item() > 0
                and self.turn_stack[idx, -1].item() == 0
            ):
                return idx
        msg = 'No attack card found on the table'
        raise ValueError(msg)

    def is_game_over(self) -> bool:
        return self.phase == GamePhase.END_GAME


class DurakEnv(Env):
    def __init__(self, game: DurakTensorGame) -> None:
        super().__init__()
        self.game = game

        # Определяем пространство действий (индексы 0-37)
        self.action_space = spaces.Discrete(38)

        # Определяем пространство состояний:
        # - Рука игрока: 36
        # - Стол: 36 * 37
        # - Сброс: 36
        # - Козырь: 4
        # - Фаза игры: 5 (one-hot encoding)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(36 + 36 * 37 + 36 + 4 + 5,), dtype=float
        )

    def add_player(self, player: Player) -> None:
        self.game.players.append(player)

    def initialize_game(self) -> None:
        self.game.initialize_game()

    def _get_observation(self) -> torch.Tensor:
        """
        Создаёт текущее состояние игры в виде одномерного тензора.
        """
        attacker = self.game.current_attacker
        defender = self.game.current_defender

        # Собираем состояния: рука атакующего, стол, сброс, козырь, фаза
        hand_tensor = attacker.cards
        table_tensor = self.game.turn_stack.flatten()
        discard_tensor = self.game.discard_pile
        trump_tensor = torch.zeros(4)
        trump_tensor[self.game.trump_suit] = 1.0

        # One-hot фаза игры
        phase_tensor = torch.zeros(5)
        phase_index = {
            GamePhase.ATTACK: 0,
            GamePhase.DEFENSE: 1,
            GamePhase.END_TURN: 2,
            GamePhase.ROTATE: 3,
            GamePhase.END_GAME: 4,
        }[self.game.phase]
        phase_tensor[phase_index] = 1.0

        # Объединяем всё в один вектор
        return torch.cat([
            hand_tensor,
            table_tensor,
            discard_tensor,
            trump_tensor,
            phase_tensor,
        ])

    def reset(
        self,
        seed: int | None = None,
        options: None = None,  # we don't have options
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Сбрасывает игру к начальному состоянию.
        """
        super().reset(seed=seed)
        self.game = DurakTensorGame()
        self.game.initialize_game()
        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """
        Выполняет шаг в игре.
        :param action: Индекс действия (0-37).
        """
        action_tensor = torch.zeros(38)
        action_tensor[action] = 1.0

        try:
            self.game.take_step(action_tensor)
        except ValueError as e:
            # Возвращаем штраф за некорректное действие
            return self._get_observation(), -1.0, False, {'error': str(e)}

        # Вычисляем награду (например, за успешное действие)
        reward = self._calculate_reward()

        # Проверяем окончание игры
        done = self.game.is_game_over()

        # Возвращаем новое состояние
        return self._get_observation(), reward, done, {}

    def _calculate_reward(self) -> float:
        """
        Вычисляет награду для текущего шага.
        """
        # Пример: вознаграждаем успешную защиту или атаку
        if self.game.phase == GamePhase.DEFENSE:
            return 1.0  # Успешная защита
        elif self.game.phase == GamePhase.END_TURN:
            return 2.0  # Завершение хода
        return 0.0  # Остальные действия

    def render(self, mode: str = 'human') -> None:
        """
        Визуализация текущего состояния игры.
        """
        print('Current Phase:', self.game.phase)
        print(
            'Attacker Hand:',
            self.game.current_attacker.cards.nonzero(as_tuple=True),
        )
        print(
            'Defender Hand:',
            self.game.current_defender.cards.nonzero(as_tuple=True),
        )
        print('Trump Suit:', self.game.trump_suit)
        print('Turn Stack:\n', self.game.turn_stack)
        print('Discard Pile:', self.game.discard_pile.nonzero(as_tuple=True))


class Agent:
    def __init__(
        self, n_observations: int, n_actions: int, memory_capacity: int
    ) -> None:
        # Сеть для текущей политики
        self.policy_net = DQN(n_observations, n_actions).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        # Целевая сеть
        self.target_net = DQN(n_observations, n_actions).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Целевая сеть не обучается

        # Replay memory
        self.memory = ReplayMemory(memory_capacity)

        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True
        )

        # Параметры стратегии ε-greedy
        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Выбор действия с использованием ε-greedy стратегии.
        """
        global steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # Выбор действия с максимальным Q-значением
                return self.policy_net(state).argmax(dim=1).view(1, 1)
        else:
            # Случайное действие
            return torch.tensor(
                [[random.randrange(OUTPUT_LEN)]], dtype=torch.long
            )

    def optimize_model(self) -> float | None:
        """
        Обновление policy_net с использованием градиентного спуска.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Вычисляем state-action pair
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Вычисляем Q-значения для текущих состояний
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # Вычисляем Q-значения для следующих состояний
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            expected_state_action_values = (
                next_state_values * GAMMA
            ) + reward_batch

        # Вычисляем функцию потерь
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Обновляем веса сети
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        """
        Обновление target_net с использованием soft update (TAU).
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - TAU) + policy_param.data * TAU
            )


# ================================
class RandomAgent:
    def __init__(self, action_space: spaces.Discrete):
        self.action_space = action_space

    def act(self, state: torch.Tensor) -> int:
        """
        Случайное действие.
        """
        return self.action_space.sample()


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


env = DurakEnv(DurakTensorGame())

# Создание агента и случайного противника
agent = Agent(
    n_observations=env.observation_space.shape[0],
    n_actions=env.action_space.n,
    memory_capacity=10000,
)

random_agent = RandomAgent(env.action_space)

env.add_player(agent)
env.add_player(random_agent)
env.initialize_game()

# Параметры обучения
NUM_EPISODES = 1000

# Цикл обучения
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        # Атакующий агент
        if env.game.phase == GamePhase.ATTACK:
            # Действие агента
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = agent.select_action(state_tensor)
        else:
            # Действие случайного агента
            action = random_agent.act(state)

        # Выполняем шаг в окружении
        next_state, reward, done, info = env.step(action.item())

        # Сохраняем переход для агента
        if env.game.phase == GamePhase.ATTACK:
            agent.memory.push(
                Transition(
                    state=torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    action=action,
                    next_state=torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                    reward=torch.tensor([reward], dtype=torch.float32),
                    done=done,
                )
            )

        # Оптимизация модели
        loss = agent.optimize_model()

        # Обновляем состояние
        state = next_state

    # Обновление целевой сети каждые TARGET_UPDATE эпизодов
    if episode % TARGET_UPDATE == 0:
        agent.update_target_net()

    # Вывод статистики обучения
    if episode % 10 == 0:
        print(f"Episode {episode}/{NUM_EPISODES} completed.")
