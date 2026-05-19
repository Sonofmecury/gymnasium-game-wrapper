"""Evaluate a random Signal Collector policy by difficulty stage."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_collector import SignalCollectorEnv


def main() -> None:
    env = SignalCollectorEnv(seed=11)
    episodes = 24
    rewards_by_stage: dict[str, list[float]] = defaultdict(list)
    completion_by_stage: dict[str, list[bool]] = defaultdict(list)

    for _ in range(episodes):
        _, info = env.reset()
        stage = info["difficulty"]
        total_reward = 0.0

        while True:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break

        rewards_by_stage[stage].append(total_reward)
        completion_by_stage[stage].append(info["signals_remaining"] == 0)

    print(f"Random policy evaluation over {episodes} episodes")
    print("difficulty  episodes  avg_reward  completion_rate")
    print("-" * 52)
    for stage, rewards in rewards_by_stage.items():
        print(
            f"{stage:<10} {len(rewards):>8} {mean(rewards):>11.2f} "
            f"{mean(completion_by_stage[stage]):>16.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
