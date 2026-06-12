"""Unified controller wrapper for Stable-Baselines3 RL policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC


AlgorithmName = Literal["ddpg", "ppo", "sac"]


_ALGORITHMS = {
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
}


@dataclass
class RLActionSnapshot:
    """Small diagnostic record for one policy inference step."""

    algorithm: AlgorithmName
    action: float
    deterministic: bool


class RLController:
    """Load a trained SB3 model and expose a small controller-style API.

    The project environment uses one continuous action in ``[-1, 1]``. This
    wrapper keeps evaluation code independent from the specific SB3 algorithm
    class used to train the policy.
    """

    def __init__(
        self,
        model_path: str | Path,
        algorithm: AlgorithmName = "sac",
        *,
        deterministic: bool = True,
        device: str = "auto",
        custom_objects: dict[str, Any] | None = None,
    ) -> None:
        algorithm = algorithm.lower()
        if algorithm not in _ALGORITHMS:
            supported = ", ".join(sorted(_ALGORITHMS))
            raise ValueError(f"unsupported RL algorithm {algorithm!r}; expected one of: {supported}")

        self.algorithm: AlgorithmName = algorithm  # type: ignore[assignment]
        self.model_path = Path(model_path)
        self.deterministic = bool(deterministic)
        self.model = _ALGORITHMS[self.algorithm].load(
            self.model_path,
            device=device,
            custom_objects=custom_objects,
        )
        self.last_snapshot: RLActionSnapshot | None = None

    def reset(self) -> None:
        """Reset transient controller state."""

        self.last_snapshot = None

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool | None = None,
    ) -> tuple[np.ndarray, RLActionSnapshot]:
        """Return an action compatible with the double-cartpole environment."""

        use_deterministic = self.deterministic if deterministic is None else deterministic
        obs = np.asarray(observation, dtype=np.float32)
        action, _state = self.model.predict(obs, deterministic=use_deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if action.shape != (1,):
            raise ValueError(f"expected a single continuous action, got shape {action.shape}")

        snapshot = RLActionSnapshot(
            algorithm=self.algorithm,
            action=float(action[0]),
            deterministic=bool(use_deterministic),
        )
        self.last_snapshot = snapshot
        return action, snapshot

    @classmethod
    def load_sac(
        cls,
        model_path: str | Path,
        *,
        deterministic: bool = True,
        device: str = "auto",
    ) -> "RLController":
        """Convenience constructor for SAC policies."""

        return cls(model_path, "sac", deterministic=deterministic, device=device)

    @classmethod
    def load_ddpg(
        cls,
        model_path: str | Path,
        *,
        deterministic: bool = True,
        device: str = "auto",
    ) -> "RLController":
        """Convenience constructor for DDPG policies."""

        return cls(model_path, "ddpg", deterministic=deterministic, device=device)
