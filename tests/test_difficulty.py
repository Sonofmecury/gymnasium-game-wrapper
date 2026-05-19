from __future__ import annotations

from pathlib import Path

from signal_collector import SignalCollectorEnv
from signal_collector.config import load_difficulty_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "difficulty.yaml"


def test_difficulty_config_loads_stages() -> None:
    config = load_difficulty_config(CONFIG_PATH)

    assert config.grid_size == 8
    assert [stage.name for stage in config.stages] == [
        "easy",
        "medium",
        "hard",
        "expert",
    ]


def test_difficulty_changes_across_episodes() -> None:
    env = SignalCollectorEnv(seed=5)
    stages = []

    for _ in range(16):
        _, info = env.reset()
        stages.append(info["difficulty"])

    assert stages[0] == "easy"
    assert stages[5] == "medium"
    assert stages[10] == "hard"
    assert stages[15] == "expert"
