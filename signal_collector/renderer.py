"""Terminal renderer for Signal Collector."""

from __future__ import annotations

from signal_collector.game import Position


class TerminalRenderer:
    """Render Signal Collector as a small ASCII grid."""

    def render(
        self,
        *,
        grid_size: int,
        agent_position: Position,
        signal_positions: set[Position],
        hazard_positions: set[Position],
        difficulty_name: str,
        remaining_steps: int,
    ) -> str:
        rows: list[str] = []
        for row in range(grid_size):
            cells: list[str] = []
            for col in range(grid_size):
                position = (row, col)
                if position == agent_position:
                    cells.append("A")
                elif position in signal_positions:
                    cells.append("S")
                elif position in hazard_positions:
                    cells.append("H")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        header = f"difficulty={difficulty_name} remaining_steps={remaining_steps}"
        return "\n".join([header, *rows])
