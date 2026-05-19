"""Run a random policy in Signal Collector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_collector import SignalCollectorEnv


def main() -> None:
    env = SignalCollectorEnv(render_mode="ansi", seed=4)

    for episode in range(4):
        _, info = env.reset()
        total_reward = 0.0
        print(f"\nEpisode {episode} difficulty={info['difficulty']}")

        while True:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break

        print(env.render())
        print(
            f"finished steps={info['step_count']} reward={total_reward:.2f} "
            f"signals_remaining={info['signals_remaining']}"
        )

    env.close()


if __name__ == "__main__":
    main()
