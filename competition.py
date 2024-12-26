import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from durak_env import DurakAEC  # ваша среда
from durak_env import get_rank, get_suit, card_can_beat, is_rank_in_play
from durak_valid_model import DQNNetwork  # ваша DQN-сеть
from durak_valid_model import (
    flatten_observation,
)  # утилита для преобразования obs


# -------------------------------------------------------------------
# 1) valid_actions: определить валидные ходы для атакующего/защитника
# -------------------------------------------------------------------
def valid_actions(env: DurakAEC, agent: str) -> list[int]:
    """
    Возвращает список допустимых (валидных) действий [0..37] для данного агента.
    Исходит из логики DurakAEC:
    - Если агент терминальный (terminations / truncations) => []
    - Если атакующий (env.is_attacker[agent]==True):
        * Можно подкинуть карту (0..35), если:
          -- она есть в руке,
          -- если стол пуст или is_rank_in_play(c, env.cards_on_table) == True
        * Можно "бито" (37).
        * Обычно "взять" (36) для атакующего не разрешено (игнорируется).
    - Если защитник:
        * Можно побить карту (0..35), если:
          -- она есть в руке,
          -- card_can_beat(c, env.trump_suit, env.attacking_card) == True
          -- env.attacking_card не None
        * Можно "взять" (36).
        * Обычно "бито" (37) для защитника в вашем DurakAEC недопустимо.
          Если разрешено, добавьте 37 в список.
    """
    if env.terminations[agent] or env.truncations[agent]:
        return []

    hand = env.hands[agent]  # множество карт (int) в руке
    is_attacker = env.is_attacker[agent]
    attacking_card = env.attacking_card

    va = []
    if is_attacker:
        # (А) Карты 0..35, которые можно подкинуть
        if len(env.cards_on_table) == 0:
            # стол пуст => любые карты из руки
            va.extend(list(hand))
        else:
            # стол не пуст => только те, чей rank есть на столе
            for c in hand:
                if is_rank_in_play(c, env.cards_on_table):
                    va.append(c)
        # (Б) "бито" (37)
        va.append(37)
        # (В) "взять" (36) обычно не нужно атакующему
    else:
        # Защитник
        if attacking_card is not None:
            for c in hand:
                if card_can_beat(c, env.trump_suit, attacking_card):
                    va.append(c)
        # "взять" (36) всегда разрешено защитнику
        va.append(36)
        # "бито" (37) для защитника обычно Invalid в вашем DurakAEC.
        # Если хотите разрешить: va.append(37)

    return va


# -------------------------------------------------------------------
# 2) Функции для агентов
# -------------------------------------------------------------------

# 2.1 Жадный агент


def card_strength(card_id: int, trump_suit: int) -> int:
    """
    'Сила' карты: rank + 9, если козырь, иначе rank.
    (get_rank(card) => 0..8, где 0=6,1=7,...8=A)
    """
    r = get_rank(card_id)
    s = get_suit(card_id)
    base = r
    if s == trump_suit:
        base += 9
    return base


def greedy_action(env: DurakAEC, agent: str) -> int | None:
    """
    Жадный агент:
      - Получаем список valid_actions(...) => только разрешённые ходы
      - Из них оставляем те, которые являются картами (0..35).
        Если есть такие, берём минимальную по card_strength.
        Иначе, если в валидных есть спец-действие (36 или 37), выбираем одно из них.
      - Если valid_actions пуст, возвращаем None.
    """
    if env.terminations[agent] or env.truncations[agent]:
        return None

    va = valid_actions(env, agent)
    if not va:
        return None

    trump_suit = env.trump_suit
    card_actions = [a for a in va if 0 <= a <= 35]
    special_actions = [a for a in va if a in [36, 37]]

    if card_actions:
        # Выбираем карту с минимальной силой
        best_card = min(
            card_actions, key=lambda c: card_strength(c, trump_suit)
        )
        return best_card
    else:
        # Нет карт => выбираем любое доступное спец-действие (36 / 37)
        return special_actions[0] if special_actions else None


# 2.2 Рандомный агент


def random_action(env: DurakAEC, agent: str) -> int | None:
    """
    Случайный агент, выбирает *только из valid_actions*.
    """
    if env.terminations[agent] or env.truncations[agent]:
        return None

    va = valid_actions(env, agent)
    if not va:
        return None

    return random.choice(va)


# 2.3 Модель (пока без маски) — можно добавить маскировку по аналогии, если хотите
def model_action(
    policy_net: nn.Module, env: DurakAEC, agent: str, device: torch.device
) -> int | None:
    """
    Действие по загруженной модели.
    Сейчас выбираем argmax Q(a) по всем 38,
    НЕ исключая недопустимых (может быть invalid action).
    Для устранения Invalid action — нужно тоже применять valid_actions + action mask.
    """
    if env.terminations[agent] or env.truncations[agent]:
        return None

    obs = flatten_observation(env.observe(agent))
    state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    was_training = policy_net.training
    policy_net.eval()
    with torch.no_grad():
        q_values = policy_net(state_t)  # shape=[1,38]
    if was_training:
        policy_net.train()

    action = int(torch.argmax(q_values, dim=1).item())
    return action


# -------------------------------------------------------------------
# 3) Функции для запуска партий
# -------------------------------------------------------------------


def play_match(
    env: DurakAEC, policy_net: nn.Module, opponent_type: str, render=True
) -> bool:
    """
    Запускает одну партию (один эпизод).
    'player_0' использует модель -> model_action(...)
    'player_1' использует opponent_type: 'model', 'random', 'greedy'.

    Возвращает True, если 'player_0' выиграл (reward>0).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env.reset()
    env.start()

    while True:
        agent = env.agent_selection

        if render:
            env.render()

        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            if agent == 'player_0':
                # действие нейросети
                action = model_action(policy_net, env, agent, device)
            else:
                # действие противника
                if opponent_type == 'model':
                    action = model_action(policy_net, env, agent, device)
                    pass

                elif opponent_type == 'random':
                    action = random_action(env, agent)
                    while action not in valid_actions(env, agent):
                        action = model_action(policy_net, env, agent, device)
                else:  # 'greedy'
                    action = greedy_action(env, agent)

            # Если None => env.step(None)
            env.step(action)

        # Проверяем завершение эпизода
        if all(env.terminations[a] or env.truncations[a] for a in env.agents):
            break

    return env.rewards['player_0'] > 0


def evaluate_agent(n_episodes=100, opponent_type='model', render=True):
    """
    Играем n_episodes партий против указанного opponent_type.
    Считаем winrate для player_0 (модели).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = DurakAEC()

    # Загрузите вашу обученную сеть
    policy_net = DQNNetwork(state_dim=113, action_dim=38).to(device)
    policy_net.load_state_dict(
        torch.load('durak_dqn_ckpt_500.pth', map_location=device)
    )
    policy_net.eval()

    wins = 0
    for _ in tqdm(range(n_episodes)):
        did_win = play_match(env, policy_net, opponent_type, render=render)
        if did_win:
            wins += 1

    winrate = wins / n_episodes
    print(f'Opponent={opponent_type} | Winrate of player_0: {winrate:.2f}')


if __name__ == '__main__':
    # 1) self-play
    print('=== EVALUATE: SELF-PLAY ===')
    evaluate_agent(n_episodes=100, opponent_type='model', render=True)

    # 2) random opponent
    # print('=== EVALUATE: VS RANDOM ===')
    # evaluate_agent(n_episodes=100, opponent_type='random', render=True)

    # 3) greedy opponent
    # print('=== EVALUATE: VS GREEDY ===')
    # evaluate_agent(n_episodes=100, opponent_type='greedy', render=True)
