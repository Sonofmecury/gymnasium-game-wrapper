from __future__ import annotations

import pytest

from signal_collector.config import DifficultyStage
from signal_collector.game import SignalCollectorGame


def make_stage() -> DifficultyStage:
    return DifficultyStage(
        name="test",
        starts_at_episode=0,
        signal_count=1,
        hazard_count=1,
        max_steps=10,
        signal_reward=1.0,
        hazard_penalty=-0.8,
        step_penalty=-0.01,
        completion_bonus=1.5,
        moving_hazards=False,
    )


def test_signal_collection_reward_and_completion() -> None:
    game = SignalCollectorGame(grid_size=4, seed=1)
    game.reset(make_stage())
    game.agent_position = (0, 0)
    game.signal_positions = {(0, 1)}
    game.hazard_positions = set()

    step = game.step(3)

    assert step.collected_signal
    assert step.completed
    assert step.reward == pytest.approx(2.49)


def test_hazard_penalty() -> None:
    game = SignalCollectorGame(grid_size=4, seed=1)
    game.reset(make_stage())
    game.agent_position = (0, 0)
    game.signal_positions = {(3, 3)}
    game.hazard_positions = {(0, 1)}

    step = game.step(3)

    assert step.hit_hazard
    assert step.reward == pytest.approx(-0.81)
