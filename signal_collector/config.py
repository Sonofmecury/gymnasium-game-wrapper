"""Difficulty configuration loading for Signal Collector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DifficultyStage:
    """One adaptive difficulty stage for the game."""

    name: str
    starts_at_episode: int
    signal_count: int
    hazard_count: int
    max_steps: int
    signal_reward: float
    hazard_penalty: float
    step_penalty: float
    completion_bonus: float
    moving_hazards: bool


@dataclass(frozen=True)
class DifficultyConfig:
    """Validated difficulty schedule."""

    grid_size: int
    stages: tuple[DifficultyStage, ...]

    def stage_for_episode(self, episode_index: int) -> DifficultyStage:
        active = self.stages[0]
        for stage in self.stages:
            if episode_index >= stage.starts_at_episode:
                active = stage
            else:
                break
        return active


def load_difficulty_config(path: str | Path) -> DifficultyConfig:
    """Load difficulty settings from YAML."""

    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    stages = tuple(
        sorted((_parse_stage(item) for item in raw["difficulties"]), key=lambda stage: stage.starts_at_episode)
    )
    if not stages or stages[0].starts_at_episode != 0:
        raise ValueError("At least one difficulty stage must start at episode 0")

    return DifficultyConfig(grid_size=int(raw["grid_size"]), stages=stages)


def _parse_stage(item: dict[str, Any]) -> DifficultyStage:
    return DifficultyStage(
        name=str(item["name"]),
        starts_at_episode=int(item["starts_at_episode"]),
        signal_count=int(item["signal_count"]),
        hazard_count=int(item["hazard_count"]),
        max_steps=int(item["max_steps"]),
        signal_reward=float(item["signal_reward"]),
        hazard_penalty=float(item["hazard_penalty"]),
        step_penalty=float(item["step_penalty"]),
        completion_bonus=float(item["completion_bonus"]),
        moving_hazards=bool(item.get("moving_hazards", False)),
    )
