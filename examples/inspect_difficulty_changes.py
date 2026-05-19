"""Print how difficulty changes across episodes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_collector import SignalCollectorEnv


def main() -> None:
    env = SignalCollectorEnv(seed=7)
    previous = None

    for _ in range(18):
        _, info = env.reset()
        stage = info["difficulty"]
        marker = "transition" if stage != previous else "active"
        print(
            f"episode={info['episode']:02d} difficulty={stage:<7} "
            f"status={marker:<10} hazards={len(info['hazards'])} "
            f"signals={info['signals_remaining']}"
        )
        previous = stage

    env.close()


if __name__ == "__main__":
    main()
