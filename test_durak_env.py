from typing import TextIO

import pytest
from gymnasium.spaces import Discrete, Dict

from notebook_script import DurakAEC


@pytest.fixture
def env() -> DurakAEC:
    """A pytest fixture that creates and returns a fresh environment."""
    environment = DurakAEC(seed=42)
    return environment


def test_env_init(env: DurakAEC) -> None:
    """
    Test that environment initializes properly:
    - possible_agents
    - action_spaces
    - observation_spaces
    """
    assert hasattr(env, 'possible_agents')
    assert len(env.possible_agents) == 2

    # Check that each agent has the correct action space
    for agent in env.possible_agents:
        assert agent in env.action_spaces
        assert isinstance(env.action_spaces[agent], Discrete)
        assert env.action_spaces[agent].n == 38

    # Check that each agent has an observation space
    for agent in env.possible_agents:
        assert agent in env.observation_spaces
        obs_space = env.observation_spaces[agent]
        assert isinstance(obs_space, Dict)
        assert set(obs_space.spaces.keys()) == {
            'hand',
            'cards_on_table',
            'cards_in_discard',
            'trump_suit',
            'is_attacker',
            'opp_cards_count',
            'deck_count',
            'attacking_card',
        }, 'Observation keys do not match expected structure'


def test_env_seed(env: DurakAEC) -> None:
    """
    Test that seeding works and does not crash.
    (Exact random sequences can be tested, but here we only test that it does not fail.)
    """
    env.seed(123)

    init_deck = env.deck.copy()
    # Possibly check that repeated seed leads to same deck ordering if you
    # store that info in the environment. Here just ensure no crash.
    env.seed(123)

    assert init_deck == env.deck


def test_env_reset(env: DurakAEC) -> None:
    """
    Check that reset:
    - does not crash
    - sets all agents as non-terminated
    - sets rewards to zero
    """
    env.reset()
    for agent in env.possible_agents:
        assert env.terminations[agent] is False
        assert env.truncations[agent] is False
        assert env.rewards[agent] == 0.0


def test_env_start(env: DurakAEC) -> None:
    """
    Check that calling `start()` does not crash.
    In your actual environment, you might verify that
    the initial 6 cards are dealt, trump suit is set, etc.
    """
    env.start()
    for agent in env.possible_agents:
        env.observe(agent)[0]


def test_observe(env: DurakAEC) -> None:
    """
    Check that observe(agent) returns a dict matching the observation space.
    """
    env.reset()
    env.start()
    agent = env.agent_selection
    obs = env.observe(agent)

    # Basic check that the returned observation has correct keys
    expected_keys = {
        'hand',
        'cards_on_table',
        'cards_in_discard',
        'trump_suit',
        'is_attacker',
        'opp_cards_count',
        'deck_count',
        'attacking_card',
    }
    assert set(obs.keys()) == expected_keys

    # Check shapes
    assert obs['hand'].shape == (36,)
    assert obs['cards_on_table'].shape == (36,)
    assert obs['cards_in_discard'].shape == (36,)
    # Others are scalars/Discrete or single integers
    # We won't check exact values here, just that no error occurs.


def test_single_step(env: DurakAEC) -> None:
    """
    Perform a single step with a random action for the
    current agent. Check that it doesn't crash and
    that step can be called repeatedly.
    """
    env.reset()
    env.start()

    for _ in range(3):
        agent = env.agent_selection
        if agent is None:
            # all agents done
            break
        action_space = env.action_spaces[agent]
        random_action = action_space.sample()
        env.step(random_action)


def test_run_until_done(env: DurakAEC) -> None:
    """
    Run the environment in a loop with random actions
    until all agents are done (terminations).
    Ensures the environment eventually finishes.
    """
    env.reset()
    env.start()

    num_steps = 0
    max_steps = 100  # just to prevent infinite loops in case of a bug

    while True:
        agent = env.agent_selection
        if agent is None:
            # means all agents are terminated
            break
        action_space = env.action_spaces[agent]
        random_action = action_space.sample()
        env.step(random_action)

        num_steps += 1
        if num_steps > max_steps:
            pytest.fail(
                'Environment did not finish within 100 steps (possible infinite loop).'
            )
            break

    # By the time we exit, all should be terminated
    for a in env.possible_agents:
        assert env.terminations[a] is True


def test_render(env: DurakAEC, capsys: TextIO) -> None:
    """
    Check that render does not crash and (optionally) prints something.
    We use capsys to capture stdout, but we only check that it doesn't fail.
    """
    env.reset()
    env.start()
    env.render(mode='human')
    captured = capsys.readouterr()
    # You could optionally check something about the printed text:
    # For example: assert "Deck" in captured.out
    # But we'll just ensure no crash.


def test_multiple_resets(env: DurakAEC) -> None:
    """
    Check that calling reset multiple times in a row doesn't break anything.
    """
    for _ in range(5):
        env.reset()
        # Optionally do a single step:
        agent = env.agent_selection
        if agent is not None:
            action_space = env.action_spaces[agent]
            env.step(action_space.sample())
