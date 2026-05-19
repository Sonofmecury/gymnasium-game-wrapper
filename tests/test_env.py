from __future__ import annotations

import numpy as np
import pytest

from signal_collector import SignalCollectorEnv


def test_env_reset_returns_valid_observation() -> None:
    env = SignalCollectorEnv(seed=2)

    observation, info = env.reset()

    assert set(observation) == {
        "agent",
        "signals",
        "hazards",
        "difficulty",
        "remaining_steps",
    }
    assert observation["agent"].dtype == np.int32
    assert info["difficulty"] == "easy"


def test_step_returns_gymnasium_format() -> None:
    env = SignalCollectorEnv(seed=3)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(4)

    assert set(observation) == {
        "agent",
        "signals",
        "hazards",
        "difficulty",
        "remaining_steps",
    }
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_count"] == 1


def test_invalid_action_raises_value_error() -> None:
    env = SignalCollectorEnv(seed=4)
    env.reset()

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)
