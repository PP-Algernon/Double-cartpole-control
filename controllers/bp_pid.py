"""BP neural-network tuned PID controller for the double-cartpole environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass
class PIDSnapshot:
    """Small diagnostics payload returned by the controller."""

    error: float
    integral: float
    derivative: float
    kp: float
    ki: float
    kd: float
    action: float


class BPNetwork(nn.Module):
    """Back-propagation network that produces non-negative PID gains.

    The output is bounded with ``sigmoid`` and scaled by ``gain_limits`` so the
    online adaptive controller cannot jump to numerically dangerous gains.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        gain_limits: Iterable[float] = (5.0, 1.0, 2.0),
    ) -> None:
        super().__init__()
        self.register_buffer(
            "gain_limits", torch.as_tensor(tuple(gain_limits), dtype=torch.float32)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features)
        return torch.sigmoid(raw) * self.gain_limits.to(raw.device)


class BPPIDController:
    """Adaptive BP-PID controller for a single continuous cart force.

    The copied environment exposes observations as:

    ``[cart_velocity, pole1_angle, pole2_angle, pole1_angular_velocity,
    pole2_angular_velocity, cart_position]``

    All values are normalized to ``[-1, 1]``.  The controller first folds the
    multi-variable balancing objective into one signed error, then lets a small
    BP network produce PID gains for the current PID features.
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        hidden_dim: int = 32,
        gain_limits: Iterable[float] = (5.0, 0.6, 1.5),
        error_weights: Iterable[float] = (-0.35, 2.4, 1.6, 0.25, 0.18, -0.8),
        integral_limit: float = 2.0,
        action_limit: float = 1.0,
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
    ) -> None:
        self.dt = float(dt)
        self.error_weights = np.asarray(tuple(error_weights), dtype=np.float32)
        self.integral_limit = float(integral_limit)
        self.action_limit = float(action_limit)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.network = BPNetwork(
            input_dim=4, hidden_dim=hidden_dim, gain_limits=gain_limits
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.reset()

    def reset(self) -> None:
        self.prev_error = 0.0
        self.prev_action = 0.0
        self.integral = 0.0
        self._last_raw_action: torch.Tensor | None = None
        self._last_pid_terms: torch.Tensor | None = None
        self._last_error = 0.0
        self._last_previous_action = 0.0
        self.last_snapshot: PIDSnapshot | None = None

    def observation_error(self, observation: np.ndarray) -> float:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (6,):
            raise ValueError(f"expected observation shape (6,), got {obs.shape}")
        return float(np.dot(self.error_weights, obs))

    def act(self, observation: np.ndarray) -> tuple[np.ndarray, PIDSnapshot]:
        error = self.observation_error(observation)
        self.integral = float(
            np.clip(
                self.integral + error * self.dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative = (error - self.prev_error) / self.dt

        features = torch.tensor(
            [error, self.integral, derivative, self.prev_error],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        gains = self.network(features).squeeze(0)
        pid_terms = torch.tensor(
            [error, self.integral, derivative],
            dtype=torch.float32,
            device=self.device,
        )
        raw_action = torch.dot(gains, pid_terms)
        action_tensor = torch.tanh(raw_action) * self.action_limit
        action = float(action_tensor.detach().cpu().item())

        self._last_raw_action = raw_action
        self._last_pid_terms = pid_terms
        self._last_error = error
        self._last_previous_action = self.prev_action
        self.last_snapshot = PIDSnapshot(
            error=error,
            integral=self.integral,
            derivative=derivative,
            kp=float(gains[0].detach().cpu().item()),
            ki=float(gains[1].detach().cpu().item()),
            kd=float(gains[2].detach().cpu().item()),
            action=action,
        )

        self.prev_error = error
        self.prev_action = action
        return np.array([action], dtype=np.float32), self.last_snapshot

    def learn_from_transition(self, next_observation: np.ndarray) -> float:
        """One online BP-PID update using a classic plant-sensitivity surrogate.

        Pymunk is not differentiable, so the update uses the sign of the latest
        finite-difference plant response and backpropagates through the PID
        control law.  It is intentionally conservative; use ``adapt=True`` in
        the evaluator when you want online tuning.
        """

        if self._last_raw_action is None:
            return 0.0

        next_error = self.observation_error(next_observation)
        action_delta = (
            self.last_snapshot.action - self._last_previous_action
            if self.last_snapshot
            else 0.0
        )
        error_delta = next_error - self._last_error

        if abs(action_delta) < 1e-6:
            plant_sign = 1.0
        else:
            plant_sign = float(np.sign(error_delta / action_delta))
            if plant_sign == 0.0:
                plant_sign = 1.0

        error_tensor = torch.tensor(next_error, dtype=torch.float32, device=self.device)
        sign_tensor = torch.tensor(plant_sign, dtype=torch.float32, device=self.device)
        regularizer = 1e-4 * self._last_raw_action.pow(2)
        loss = (
            error_tensor.detach() * sign_tensor.detach() * self._last_raw_action
            + regularizer
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def save(self, path: str | Path) -> None:
        payload = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "dt": self.dt,
            "error_weights": self.error_weights.tolist(),
            "integral_limit": self.integral_limit,
            "action_limit": self.action_limit,
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        try:
            payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(Path(path), map_location=self.device)
        except Exception:
            payload = torch.load(Path(path), map_location=self.device, weights_only=False)

        self.network.load_state_dict(payload["network"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.dt = float(payload.get("dt", self.dt))
        if "error_weights" in payload:
            self.error_weights = np.asarray(payload["error_weights"], dtype=np.float32)
        self.integral_limit = float(payload.get("integral_limit", self.integral_limit))
        self.action_limit = float(payload.get("action_limit", self.action_limit))
        self.reset()
