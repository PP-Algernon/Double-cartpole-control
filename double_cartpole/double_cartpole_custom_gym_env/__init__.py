"""Gymnasium registration for the local double-cartpole environment."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from .double_cartpole_env import DoubleCartpoleEnv

ENV_ID = "double-cartpole-custom-v0"


def register_env() -> None:
    """Register the environment once for ``gymnasium.make``."""

    if ENV_ID in registry:
        return
    register(
        id=ENV_ID,
        entry_point=DoubleCartpoleEnv,
        kwargs={"render_sim": False, "n_steps": 1000},
    )


register_env()

__all__ = ["DoubleCartpoleEnv", "ENV_ID", "register_env"]
