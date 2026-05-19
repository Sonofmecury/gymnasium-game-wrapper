"""Gymnasium wrapper for the Signal Collector game."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from signal_collector.config import DifficultyConfig, DifficultyStage, load_difficulty_config
from signal_collector.game import Position, SignalCollectorGame
from signal_collector.renderer import TerminalRenderer


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "difficulty.yaml"


class SignalCollectorEnv(gym.Env):
    """Gymnasium-compatible wrapper around SignalCollectorGame."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config: DifficultyConfig = load_difficulty_config(config_path)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.episode_index = -1
        self.current_stage: DifficultyStage = self.config.stages[0]
        self.game = SignalCollectorGame(self.config.grid_size, seed=seed)
        self.renderer = TerminalRenderer()

        self.action_space = spaces.Discrete(5)
        if seed is not None:
            self.action_space.seed(seed)

        max_cells = self.config.grid_size * self.config.grid_size
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.config.grid_size - 1, shape=(2,), dtype=np.int32),
                "signals": spaces.MultiBinary(max_cells),
                "hazards": spaces.MultiBinary(max_cells),
                "difficulty": spaces.Discrete(len(self.config.stages)),
                "remaining_steps": spaces.Box(0, max(stage.max_steps for stage in self.config.stages), shape=(1,), dtype=np.int32),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.game.rng.seed(seed)

        self.episode_index += 1
        self.current_stage = self.config.stage_for_episode(self.episode_index)
        self.game.reset(self.current_stage)
        return self._observation(), self._info(last_reward=0.0)

    def step(self, action: int) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Invalid action; expected an integer in [0, 4]")

        step = self.game.step(int(action))
        return (
            self._observation(),
            float(step.reward),
            step.completed,
            step.truncated,
            self._info(
                last_reward=float(step.reward),
                collected_signal=step.collected_signal,
                hit_hazard=step.hit_hazard,
            ),
        )

    def render(self) -> str:
        frame = self.renderer.render(
            grid_size=self.config.grid_size,
            agent_position=self.game.agent_position,
            signal_positions=self.game.signal_positions,
            hazard_positions=self.game.hazard_positions,
            difficulty_name=self.current_stage.name,
            remaining_steps=self.current_stage.max_steps - self.game.step_count,
        )
        if self.render_mode == "ansi":
            return frame
        print(frame)
        return frame

    def _observation(self) -> dict[str, np.ndarray | int]:
        return {
            "agent": np.array(self.game.agent_position, dtype=np.int32),
            "signals": self._position_mask(self.game.signal_positions),
            "hazards": self._position_mask(self.game.hazard_positions),
            "difficulty": self.config.stages.index(self.current_stage),
            "remaining_steps": np.array([self.current_stage.max_steps - self.game.step_count], dtype=np.int32),
        }

    def _position_mask(self, positions: set[Position]) -> np.ndarray:
        mask = np.zeros(self.config.grid_size * self.config.grid_size, dtype=np.int8)
        for row, col in positions:
            mask[row * self.config.grid_size + col] = 1
        return mask

    def _info(self, *, last_reward: float, collected_signal: bool = False, hit_hazard: bool = False) -> dict[str, Any]:
        return {
            "episode": self.episode_index,
            "difficulty": self.current_stage.name,
            "step_count": self.game.step_count,
            "signals_remaining": len(self.game.signal_positions),
            "hazards": sorted(self.game.hazard_positions),
            "last_reward": last_reward,
            "collected_signal": collected_signal,
            "hit_hazard": hit_hazard,
        }
