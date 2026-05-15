"""Convenience wrappers for the local Gymnasium double-cartpole environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


DOUBLE_CARTPOLE_ENV_ID = "double-cartpole-custom-v0"


def register_double_cartpole() -> str:
    """Import the local package so Gymnasium sees ``DOUBLE_CARTPOLE_ENV_ID``."""

    try:
        from double_cartpole_custom_gym_env import register_env
    except ModuleNotFoundError as exc:
        if exc.name != "double_cartpole_custom_gym_env":
            raise
        from double_cartpole.double_cartpole_custom_gym_env import register_env

    register_env()

    return DOUBLE_CARTPOLE_ENV_ID


def make_double_cartpole(
    *,
    render_mode: str | None = None,
    render_sim: bool | None = None,
    n_steps: int = 1000,
    force_scale: float = 1200.0,
    **kwargs: Any,
) -> gym.Env:
    """Create the project double-cartpole environment through Gymnasium."""

    env_id = register_double_cartpole()
    return gym.make(
        env_id,
        render_mode=render_mode,
        render_sim=render_sim,
        n_steps=n_steps,
        force_scale=force_scale,
        **kwargs,
    )


register_double_cartpole()
