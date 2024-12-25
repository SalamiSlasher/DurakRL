# %%NBQA-CELL-SEP08306a
from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TypedDict

import random

from gymnasium.spaces import Box
from gymnasium.spaces import Dict
from gymnasium.spaces import Discrete
from pettingzoo.utils import AECEnv
from pettingzoo.utils import agent_selector
from collections import deque

from torch import nn
from torch import optim

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

hash(0x6BB8CEA7)


# %%NBQA-CELL-SEP08306a
class DurakObservation(TypedDict):
    hand: list[int]
    cards_on_table: list[int]
    cards_in_discard: list[int]
    trump_suit: int
    is_attacker: bool
    opp_cards_count: int
    deck_count: int
    attacking_card: int | None


# %%NBQA-CELL-SEP08306a
NUM_PLAYERS = 2
DECK_SIZE = 36
ALL_AGENTS = [f'player_{i}' for i in range(NUM_PLAYERS)]
CARD_DRAW_REWARD = 0.05


# %%NBQA-CELL-SEP08306a
suites = ['♥', '♦', '♠', '♣']
ranks = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


def get_rank(card_id: int) -> int:
    return card_id // 4


def get_suit(card_id: int) -> int:
    return card_id % 4


def pretty_card(card_id: int | None) -> str:
    return (
        ranks[get_rank(card_id)] + suites[get_suit(card_id)]
        if card_id is not None
        else 'No'
    )


# %%NBQA-CELL-SEP08306a
def card_can_beat(
    *, defend_card: int, trump_suit: int, attacking_card: int
) -> bool:
    """Check if defend_card can beat attack_card with this trump_suit."""
    attack_rank = get_rank(attacking_card)
    attack_suit = get_suit(attacking_card)

    defend_rank = get_rank(defend_card)
    defend_suit = get_suit(defend_card)

    # same suit
    if defend_suit == attack_suit and defend_rank > attack_rank:
        return True

    # only defend card suit is trump
    if defend_suit == trump_suit and attack_suit != trump_suit:
        return True

    return False


def is_rank_in_play(card_id: int, cards_on_table: list[int]) -> bool:
    """Check if card in play, to determine if attacker can play it."""
    if not cards_on_table:
        return True  # if no cards in play, anything goes
    table_ranks = {get_rank(c) for c in cards_on_table}
    return get_rank(card_id) in table_ranks


# %%NBQA-CELL-SEP08306a
type Agent = str


class DurakAEC(AECEnv):
    """
    Пример среды "Дурак" для двух игроков, с упрощённой логикой многократной атаки:
    - Атакующий может подкидывать карты (те, чей ранг есть на столе).
    - Защитник может побить или взять карты.
    - Если защитник взял карты, раунд заканчивается немедленно,
      и в следующем раунде снова ходит тот же атакующий.
    - Если атакующий сказал "бито" (37), раунд заканчивается, и роли меняются.
    - Игра завершается, когда у одного из игроков нет карт в руке и колода пуста:
      этот игрок побеждает, второй проигрывает.
    """

    metadata = {'render_modes': ['human'], 'name': 'durak_multi_attack_v0'}

    def __init__(
        self, seed: int | None = None, agents: Sequence[Agent] = ()
    ) -> None:
        """
        :param seed: для воспроизводимости
        :param agents: список внешних агентов (можете использовать для self-play или нет).
                       Здесь просто сохраним их, а в демо-коде ниже покажем, как это может выглядеть.
        """
        super().__init__()

        self.seed_value = seed  # Для воспроизводимости, если нужно фиксировать генератор случайных чисел.
        self.external_agents = list(
            agents
        )  # Сохраняем список внешних агентов, если они заданы.

        self.possible_agents = ALL_AGENTS  # Список всех возможных агентов (в данном случае два игрока).
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }  # Сопоставление имени агента с индексом.

        # Определяем пространство действий (всего 38 возможных действий: сыграть карту, взять, или "бито").
        self.action_spaces = {
            agent: Discrete(
                DECK_SIZE + 2
            )  # 0..35 для карт, 36 = взять, 37 = бито.
            for agent in self.possible_agents
        }

        # Пространства наблюдений для каждого агента.
        # Используем Dict для описания сложной структуры наблюдений, соответствующих игровому состоянию.
        self.observation_spaces = {}
        for agent in self.possible_agents:
            self.observation_spaces[agent] = Dict({
                'hand': Box(
                    low=0, high=DECK_SIZE, shape=(36,), dtype=np.int32
                ),  # Максимум 36 карт в руке.
                'cards_on_table': Box(
                    low=0, high=DECK_SIZE, shape=(36,), dtype=np.int32
                ),  # Карты на столе.
                'cards_in_discard': Box(
                    low=0, high=DECK_SIZE, shape=(36,), dtype=np.int32
                ),  # Сброшенные карты.
                'trump_suit': Discrete(
                    4
                ),  # Масть козыря: 0 = черви, 1 = бубны, 2 = трефы, 3 = пики.
                'is_attacker': Discrete(
                    2
                ),  # Флаг: атакует ли данный агент (1 = да, 0 = нет).
                'opp_cards_count': Box(
                    low=0, high=36, shape=(), dtype=np.int32
                ),  # Количество карт у соперника.
                'deck_count': Box(
                    low=0, high=36, shape=(), dtype=np.int32
                ),  # Количество карт в колоде.
                'attacking_card': Box(
                    low=0, high=DECK_SIZE, shape=(), dtype=np.int32
                ),  # ID атакующей карты (или None).
            })
        self.reset()

    def seed(self, seed=None) -> None:
        """Устанавливаем seed для рандома."""
        if seed is not None:
            self.seed_value = seed
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)

    def reset(self, seed=None, options=None) -> None:
        self.seed(
            seed if seed is not None else self.seed_value
        )  # set seed if present

        self.agents = self.possible_agents[:]  # copying agents list
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Перемешиваем колоду
        self.deck = list(range(DECK_SIZE))
        random.shuffle(self.deck)

        # Руки
        self.hands = {
            agent: [] for agent in self.agents
        }  # init empty hands for each player
        self.cards_on_table = []  # Карты на столе (упростим: будем класть все атакующие карты + бьющие)
        self.cards_in_discard = []  # Бито/сброс

        self.trump_suit = None
        self.attacking_card = (
            None  # текущая атакующая карта (если идёт защита прямо сейчас)
        )
        type boolshit = bool
        # bool: True => агент является атакующим
        self.is_attacker: dict[Agent, boolshit] = dict.fromkeys(
            self.agents, False
        )  # {'player1': False, 'player2': False}

        # Флаги окончания
        self.terminations: dict[Agent, bool] = dict.fromkeys(self.agents, False)
        self.truncations: dict[Agent, bool] = dict.fromkeys(self.agents, False)
        self.rewards: dict[Agent, float] = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards: dict[Agent, float] = {
            agent: 0.0 for agent in self.possible_agents
        }
        self.infos: dict[Agent, dict] = {agent: {} for agent in self.agents}

        self.game_started = False
        self.major_turn = 0
        self.minor_turn = 0
        self.last_action: int | None = None, None

    def start(self):
        """Старт игры: раздаём по 6 карт, определяем козырь, выбираем первого атакующего."""
        last_card = 0
        for i in range(6):  # Раздаём каждому агенту по 6 карт.
            for agent in self.agents:
                if self.deck:
                    last_card = self.deck.pop()
                    self.hands[agent].append(
                        last_card
                    )  # Берём карту из колоды и добавляем в руку.

        if (
            self.deck
        ):  # Если колода не пуста, определяем козырь по верхней карте.
            top_card = self.deck[-1]
            self.trump_suit = get_suit(
                top_card
            )  # Масть верхней карты задаёт козырь.
        else:
            self.trump_suit = get_suit(
                last_card
            )  # Если колода пуста, по козырь - последняя карта.

        # Определяем, у кого минимальная козырная карта
        min_trump_card_per_agent: dict[
            Agent, int | None
        ] = {}  # Словарь для минимальной козырной карты каждого агента.
        for agent in self.agents:
            trumps = [
                c for c in self.hands[agent] if get_suit(c) == self.trump_suit
            ]  # Находим все козыри в руке агента.
            if trumps:
                min_trump_card_per_agent[agent] = min(
                    trumps
                )  # Минимальный козырь, если есть.
            else:
                min_trump_card_per_agent[agent] = (
                    None  # Если козырей нет, None.
                )

        # Если у обоих None => ходит второй
        if (
            min_trump_card_per_agent[self.agents[0]] is None
            and min_trump_card_per_agent[self.agents[1]] is None
        ):  # Если у обоих агентов нет козырей, первый ходит второй агент.
            first_agent = self.agents[1]
        else:
            not_none = [
                (agent, card)
                for agent, card in min_trump_card_per_agent.items()
                if card is not None
            ]  # Список агентов с козырями.
            not_none.sort(
                key=lambda x: x[1]
            )  # Сортируем по значению минимального козыря.
            first_agent = (
                not_none[0][0] if not_none else self.agents[1]
            )  # Агент с минимальным козырём ходит первым.

        self.is_attacker[first_agent] = (
            True  # turn on one of players attack flags
        )

        self._agent_selector = agent_selector(
            [first_agent] + [a for a in self.agents if a != first_agent]
        )
        self.agent_selection = self._agent_selector.reset()

        self.game_started = True

    def observe(self, agent: Agent) -> DurakObservation:
        opp_agent = next(a for a in self.agents if a != agent)

        obs: DurakObservation = {
            'hand': sorted(self.hands[agent]),
            'cards_on_table': sorted(self.cards_on_table),
            'cards_in_discard': sorted(self.cards_in_discard),
            'trump_suit': self.trump_suit,
            'is_attacker': self.is_attacker[agent],
            'opp_cards_count': len(self.hands[opp_agent]),
            'deck_count': len(self.deck),
            'attacking_card': self.attacking_card,
        }
        return obs

    def _finish_step(self, force_next: Agent | None = None) -> None:
        """Переходим к следующему агенту (или к агенту из force_next, если хотим)."""
        if force_next is not None:
            self.agent_selection = force_next
        else:
            self.agent_selection = self._agent_selector.next()

        # Если следующий агент уже terminated, идём дальше
        while (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ) and self.agents:
            self.agent_selection = self._agent_selector.next()

        self.minor_turn += 1

    def _round_end(self, defender_took: bool):
        """
        Завершение раунда: добор карт, смена (или не смена) роли.
        """
        # Упорядочиваем игроков: сначала атакующие, затем защищающийся
        order = [agent for agent in self.agents if self.is_attacker[agent]] + [
            agent for agent in self.agents if not self.is_attacker[agent]
        ]

        # Добираем карты для каждого агента в порядке
        for agent in order:
            while len(self.hands[agent]) < 6 and self.deck:
                self.hands[agent].append(self.deck.pop())
                self.rewards[agent] += CARD_DRAW_REWARD

        if defender_took:
            # Если защитник взял карты, атакующие остаются теми же
            pass
        else:
            # Если защитник отбился, меняем роли атакующих и защищающихся
            for agent in self.agents:
                self.is_attacker[agent] = not self.is_attacker[agent]

        # Сбрасываем карты со стола в сброс
        self.cards_in_discard.extend(self.cards_on_table)
        self.cards_on_table.clear()
        self.attacking_card = None

        self.major_turn += 1
        self.minor_turn = 0

    def _check_game_done(self) -> bool:
        """
        Проверяет, завершилась ли игра. Если да, обновляет terminations и назначает финальные награды.
        """
        active_agents = [a for a in self.agents if not self.terminations[a]]

        # Проверяем, кто выигрывает, если карты закончились
        for agent in active_agents:
            if len(self.hands[agent]) == 0 and len(self.deck) == 0:
                # Победитель
                self.terminations[agent] = True
                self.rewards[agent] += 1.0  # Награда победителю

        # Если остался только один активный агент, он проигрывает
        active_agents = [a for a in self.agents if not self.terminations[a]]
        if len(active_agents) == 1:
            loser = active_agents[0]
            self.terminations[loser] = True
            self.rewards[loser] -= 1.0  # Штраф проигравшему

            # Остальные (победители) получают награды, если ещё не начислены
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.rewards[agent] += 1.0
            return True  # Игра завершена

        return False  # Игра продолжается

    def _attacker_step(self, action: int) -> None:
        agent = self.agent_selection
        opp_agent = next(a for a in self.agents if a != agent)
        # Ход атакующего
        if action < 36:
            # Проверяем, есть ли карта в руке
            if action in self.hands[agent]:
                # Проверим, можем ли мы "подкинуть" эту карту
                # Упрощённая логика: разрешаем ходить любой картой, если стол пуст
                # или ранг есть на столе, если не пуст.
                if is_rank_in_play(action, self.cards_on_table):
                    # Атакуем
                    self.hands[agent].remove(action)
                    self.cards_on_table.append(action)
                    # Теперь атакующая карта = последняя
                    self.attacking_card = action
                    # Передаём ход защитнику
                    self._finish_step(force_next=opp_agent)
                else:
                    # Недопустимый ход. Можно выдать штраф или проигнорировать.
                    # Для упрощения — просто игнорируем.
                    self._finish_step()
            else:
                self.last_action = None, agent
                self._finish_step()
        elif action == 36:
            # 36="взять" для атакующего не имеет смысла, игнорируем
            self._finish_step()
        elif action == 37:
            # "Бито" — атакующий говорит, что подкидывать больше не будет,
            # значит раунд закончился => переносим все карты со стола в discard,
            # меняем роли
            self._round_end(defender_took=False)
            self._finish_step()
        else:
            self._finish_step()

    def step(self, action: int) -> None:
        """Одновременная логика "многоходовой" атаки:
        - Если ход атакующего: сыграть карту (подкидываем) или 'бито' (37).
        - Если ход защитника: побить карту (если может) или 'взять' (36).
        """
        if not self.game_started:
            raise ValueError('Игра не начата, вызовите start()')

        agent = self.agent_selection
        self.last_action = action, agent
        if self.terminations[agent] or self.truncations[agent]:
            self._finish_step()
            return

        opp_agent = next(a for a in self.agents if a != agent)

        # Действие:
        # 0..35 => сыграть карту
        # 36 => взять
        # 37 => бито (пас от атакующего)
        if self.is_attacker[agent]:
            self._attacker_step(action)
        # Ход защитника
        elif action < 36:
            # Пытаемся побить атакующую карту
            if action in self.hands[agent] and self.attacking_card is not None:
                can_beat_it = card_can_beat(
                    defend_card=action,
                    trump_suit=self.trump_suit,
                    attacking_card=self.attacking_card,
                )
                if not can_beat_it:
                    # Нельзя побить — действие игнорируем
                    self.last_action = None, agent
                    self._finish_step()
                else:
                    # Успешно бьём: убираем карту из руки
                    self.hands[agent].remove(action)
                    # Ставим и атакующую, и бьющую в discard
                    self.cards_in_discard.append(self.attacking_card)
                    self.cards_in_discard.append(action)
                    # Убираем их со стола
                    if self.attacking_card in self.cards_on_table:
                        self.cards_on_table.remove(self.attacking_card)
                    self.attacking_card = None
                    # Возвращаем ход атакующему, он может подкинуть ещё
                    self._finish_step(force_next=opp_agent)
            else:
                self._finish_step()
        elif action == 36:
            # Защитник берёт карты
            # Забирает все карты со стола + attacking_card (если она там)
            self.hands[agent].extend(self.cards_on_table)
            if (self.attacking_card is not None) and (
                self.attacking_card not in self.cards_on_table
            ):
                # Если вдруг она уже убрана со стола, пропустим
                pass
            elif self.attacking_card is not None:
                self.hands[agent].append(self.attacking_card)

            self.cards_on_table.clear()
            self.attacking_card = None

            # Раунд закончен, роли не меняются (аттакующий снова ходит в следующем раунде)
            self._round_end(defender_took=True)
            self._finish_step()
        elif action == 37:
            # "Бито" от защитника не имеет смысла
            self.last_action = None, agent
            self._finish_step()
        else:
            # cannot be other than 0..37
            self.last_action = None, agent
            self._finish_step()

        self._check_game_done()

    def render(self, mode='human'):
        if mode == 'human':
            agent = self.agent_selection
            print(
                f'--- Agent turn #{self.major_turn}.{self.minor_turn} : {agent} ---'
            )
            if self.last_action[0] is None:
                print(f'Invalid action: Agent: {agent}')
                return None
            for ag in self.agents:
                print(
                    f'{ag} => hand: {[pretty_card(card) for card in self.hands[ag]]}, attacker={self.is_attacker[ag]}, '
                    f'terminated={self.terminations[ag]}'
                )
            print(
                f'Cards on table: {[pretty_card(card) for card in self.cards_on_table]}, Attacking card: {pretty_card(self.attacking_card)}'
            )
            print(
                f'Discard: {[pretty_card(card) for card in self.cards_in_discard]}, Deck: {len(self.deck)}, Trump suit: {suites[self.trump_suit]}'
            )
            if self.last_action[0] < 36:
                print(
                    f'Agent: {self.last_action[1]}. Action: {pretty_card(self.last_action[0])}'
                )
            elif self.last_action[0] == 36:
                print(f'Agent: {self.last_action[1]}. Action: Take')
            else:
                print(f'Agent: {self.last_action[1]}. Action: Bita')
            print('---')


# %%NBQA-CELL-SEP08306a
class SimpleDQN(nn.Module):
    """
    Очень простой MLP, принимающий "плоское" состояние
    и предсказывающий Q-значения для 38 действий.
    """

    def __init__(self, obs_dim: int, act_dim: int = 38):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, act_dim)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        return self.out(x)


# %%NBQA-CELL-SEP08306a
def flatten_observation(obs: DurakObservation, agent_id: int) -> np.ndarray:
    """
    Пример конвертации словаря наблюдений в плоский вектор.
    Включим туда:
      - one-hot: кто мы (agent_id)
      - наша рука (36 длина, 1/0 есть ли карта)
      - карты на столе (36 длина)
      - размер колоды, opp_cards_count
      - trump_suit (1 из 4)
      - is_attacker (0 или 1)
      - attacking_card (36 длина one-hot)
    """
    vec = []

    # one-hot код агента (2 игрока)
    agent_one_hot = [0, 0]
    agent_one_hot[agent_id] = 1
    vec.extend(agent_one_hot)

    # Наша рука
    hand_vec = [0] * DECK_SIZE
    for c in obs[0]['hand']:
        hand_vec[c] = 1
    vec.extend(hand_vec)

    # Карты на столе
    table_vec = [0] * DECK_SIZE
    for c in obs[0]['cards_on_table']:
        table_vec[c] = 1
    vec.extend(table_vec)

    # attacking_card - one-hot
    att_vec = [0] * DECK_SIZE
    if obs[0]['attacking_card'] is not None:
        att_vec[obs[0]['attacking_card']] = 1
    vec.extend(att_vec)

    # discard не будем явно включать (или можно включить)
    # trump_suit (4)
    ts = [0, 0, 0, 0]
    ts[obs[0]['trump_suit']] = 1
    vec.extend(ts)

    # is_attacker
    vec.append(1.0 if obs[0]['is_attacker'] else 0.0)

    # opp_cards_count + deck_count (2 штуки)
    vec.append(float(obs[0]['opp_cards_count']))
    vec.append(float(obs[0]['deck_count']))

    return np.array(vec, dtype=np.float32)


# %%NBQA-CELL-SEP08306a
class ReplayBuffer:
    """Простой буфер опыта для DQN."""

    def __init__(self, capacity=10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# %%NBQA-CELL-SEP08306a
def dqn_selfplay_training(num_episodes=2000, render=False) -> None:
    """
    Пример (упрощённый) цикла обучения DQN в среде DurakAEC.
    Один Q-network на двоих агентов. Мы кодируем "кто ходит" в состоянии.
    """
    env = DurakAEC(seed=42)
    env.reset()
    env.start()

    print(list(pretty_card(card) for card in env.deck))

    seed = 42
    env.seed(seed)

    # Определим размерность наблюдения:
    # По нашей функции flatten_observation:
    # 2 (agent_id one-hot) + 36 (hand) + 36 (table) + 36 (att_card) + 4 (trump_suit) + 1 (is_attacker)
    # + 2 (opp_cards_count, deck_count) = 117
    obs_dim = 117
    act_dim = 38

    policy_net = SimpleDQN(obs_dim, act_dim)
    target_net = SimpleDQN(obs_dim, act_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=50000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0  # eps-greedy
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    update_target_freq = 1000
    global_step = 0

    def select_action(state_vec: np.ndarray) -> np.ndarray:
        if random.random() < epsilon:
            return random.randint(0, act_dim - 1)
        with torch.no_grad():
            q_values = policy_net(
                torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            )
            return q_values.argmax(dim=1).item()

    for episode in tqdm(range(num_episodes), desc='episode'):
        env.reset()
        env.start()

        done = False
        episode_steps = 0

        # Словарь: agent -> его последнее состояние (для s,a,r перехода)
        last_obs = {}
        last_state_vec = {}

        while not done:
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                # Агент завершил — step(None) чтобы пропустить
                env.step(None)
            else:
                obs = env.last(observe=True)
                agent_id = env.agent_name_mapping[agent]
                state_vec = flatten_observation(obs, agent_id)

                action = select_action(state_vec)

                # Сохраняем (s) для последующего перехода
                last_obs[agent] = obs
                last_state_vec[agent] = state_vec

                env.step(action)

                # После шага мы можем собрать (s, a, r, s', done)
                reward = env.rewards[agent]
                # Определим done для этого агента как (termination or truncation)
                done_agent = env.terminations[agent] or env.truncations[agent]

                # new_obs = env.last(observe=True) -- но уже для другого агента,
                # поэтому чтобы собрать переход, мы используем "старое" состояние -> "текущее" (которое будет затем)
                # Упростим: когда ход агента закончился, можно взять next_state из перспективы этого же агента.
                # Но в AEC это tricky. Упростим и скажем, что s' = state_vec (agent) (хотя это не идеально).

                # Будем пушить сразу
                if done_agent:
                    # Если агент закончил — у него next_state = нули
                    ns_vec = np.zeros_like(state_vec)
                else:
                    # Можно либо подождать, когда снова будет ход этого агента (правильнее),
                    # либо (упрощённо) взять текущее состояние (для другого агента) как next_state.
                    # Для простоты запишем так:
                    ns_vec = state_vec

                replay_buffer.push(
                    last_state_vec[agent],
                    action,
                    reward,
                    ns_vec,
                    float(done_agent),
                )

                episode_steps += 1

                # Обновляем DQN, если в буфере достаточно переходов
                if len(replay_buffer) > batch_size:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = (
                        replay_buffer.sample(batch_size)
                    )

                    states_t = torch.tensor(states_b, dtype=torch.float32)
                    actions_t = torch.tensor(
                        actions_b, dtype=torch.int64
                    ).unsqueeze(1)
                    rewards_t = torch.tensor(
                        rewards_b, dtype=torch.float32
                    ).unsqueeze(1)
                    next_states_t = torch.tensor(
                        next_states_b, dtype=torch.float32
                    )
                    dones_t = torch.tensor(
                        dones_b, dtype=torch.float32
                    ).unsqueeze(1)

                    # Q(s,a)
                    q_values = policy_net(states_t).gather(1, actions_t)

                    # Q'(s', a') = max_a' target_net(s')  (Double DQN делать не будем, упрощённо)
                    with torch.no_grad():
                        next_q = target_net(next_states_t).max(
                            dim=1, keepdim=True
                        )[0]

                    target = rewards_t + gamma * (1 - dones_t) * next_q
                    loss = F.mse_loss(q_values, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Обновляем target_net
                    global_step += 1
                    if global_step % update_target_freq == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                # decay eps
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            done = all(
                env.terminations[a] or env.truncations[a] for a in env.agents
            )
            if render:
                env.render()

        # Пример: выводим прогресс
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}, epsilon={epsilon:.3f}')

    print('Training finished.')
    return target_net


# %%NBQA-CELL-SEP08306a
f = open('play.log', 'w', encoding='utf-8')
sys.stdout = f

dqn_selfplay_training(num_episodes=1, render=True)
