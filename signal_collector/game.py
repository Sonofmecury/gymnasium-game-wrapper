"""Core Signal Collector game logic independent of Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
import random

from signal_collector.config import DifficultyStage


Position = tuple[int, int]


@dataclass(frozen=True)
class GameStep:
    """Result of one game-engine update."""

    reward: float
    collected_signal: bool
    hit_hazard: bool
    completed: bool
    truncated: bool


class SignalCollectorGame:
    """Grid game where an agent collects signals while avoiding hazards."""

    ACTION_DELTAS: dict[int, Position] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (0, 0),
    }

    def __init__(self, grid_size: int, seed: int | None = None) -> None:
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        self.agent_position: Position = (0, 0)
        self.signal_positions: set[Position] = set()
        self.hazard_positions: set[Position] = set()
        self.step_count = 0
        self.stage: DifficultyStage | None = None

    def reset(self, stage: DifficultyStage) -> dict[str, object]:
        self.stage = stage
        self.step_count = 0
        self.agent_position = (0, 0)
        blocked = {self.agent_position}
        self.signal_positions = self._sample_positions(stage.signal_count, blocked)
        blocked = blocked | self.signal_positions
        self.hazard_positions = self._sample_positions(stage.hazard_count, blocked)
        return self.state()

    def step(self, action: int) -> GameStep:
        if self.stage is None:
            raise RuntimeError("Game must be reset before stepping")
        if action not in self.ACTION_DELTAS:
            raise ValueError("Action must be one of 0, 1, 2, 3, or 4")

        self.step_count += 1
        row_delta, col_delta = self.ACTION_DELTAS[action]
        row, col = self.agent_position
        self.agent_position = (
            max(0, min(self.grid_size - 1, row + row_delta)),
            max(0, min(self.grid_size - 1, col + col_delta)),
        )

        reward = self.stage.step_penalty
        collected_signal = self.agent_position in self.signal_positions
        hit_hazard = self.agent_position in self.hazard_positions

        if collected_signal:
            self.signal_positions.remove(self.agent_position)
            reward += self.stage.signal_reward

        if hit_hazard:
            reward += self.stage.hazard_penalty

        completed = len(self.signal_positions) == 0
        if completed:
            reward += self.stage.completion_bonus

        truncated = self.step_count >= self.stage.max_steps

        if self.stage.moving_hazards and not completed:
            self._move_hazards()

        return GameStep(
            reward=reward,
            collected_signal=collected_signal,
            hit_hazard=hit_hazard,
            completed=completed,
            truncated=truncated,
        )

    def state(self) -> dict[str, object]:
        if self.stage is None:
            remaining_steps = 0
            difficulty = "unconfigured"
        else:
            remaining_steps = max(0, self.stage.max_steps - self.step_count)
            difficulty = self.stage.name
        return {
            "agent": self.agent_position,
            "signals": sorted(self.signal_positions),
            "hazards": sorted(self.hazard_positions),
            "difficulty": difficulty,
            "remaining_steps": remaining_steps,
            "step_count": self.step_count,
        }

    def _sample_positions(self, count: int, blocked: set[Position]) -> set[Position]:
        available = [
            (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
            if (row, col) not in blocked
        ]
        if count > len(available):
            raise ValueError("Not enough free cells to sample positions")
        return set(self.rng.sample(available, count))

    def _move_hazards(self) -> None:
        occupied = {self.agent_position} | self.signal_positions
        moved: set[Position] = set()
        for hazard in sorted(self.hazard_positions):
            candidates = self._neighbor_positions(hazard)
            self.rng.shuffle(candidates)
            selected = hazard
            for candidate in candidates:
                if candidate not in occupied and candidate not in moved:
                    selected = candidate
                    break
            moved.add(selected)
        self.hazard_positions = moved

    def _neighbor_positions(self, position: Position) -> list[Position]:
        row, col = position
        candidates = []
        for row_delta, col_delta in self.ACTION_DELTAS.values():
            candidate = (
                max(0, min(self.grid_size - 1, row + row_delta)),
                max(0, min(self.grid_size - 1, col + col_delta)),
            )
            candidates.append(candidate)
        return candidates
