"""Signal Collector game and Gymnasium environment."""

from gymnasium.envs.registration import register, registry

from signal_collector.env import SignalCollectorEnv
from signal_collector.game import SignalCollectorGame

if "SignalCollector-v0" not in registry:
    register(
        id="SignalCollector-v0",
        entry_point="signal_collector.env:SignalCollectorEnv",
        max_episode_steps=200,
    )

__version__ = "0.1.0"

__all__ = ["SignalCollectorEnv", "SignalCollectorGame"]
