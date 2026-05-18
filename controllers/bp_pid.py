"""用于双倒立摆环境的BP神经网络整定PID控制器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass
class PIDSnapshot:
    """控制器返回的简短诊断信息。"""

    error: float
    integral: float
    derivative: float
    kp: float
    ki: float
    kd: float
    action: float


class BPNetwork(nn.Module):
    """反向传播网络,用于生成非负的PID增益。

    输出通过 ``sigmoid`` 函数激活，并由 ``gain_limits`` 进行缩放，
    以防止在线自适应控制器跳跃到数值不稳定的增益值。
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        gain_limits: Iterable[float] = (2.0, 0.05, 0.2),
        initial_gains: Iterable[float] = (1.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        initial_gains_tensor = torch.as_tensor(tuple(initial_gains), dtype=torch.float32)
        self.register_buffer(
            "gain_limits", torch.as_tensor(tuple(gain_limits), dtype=torch.float32)
        )
        self.register_buffer("initial_gains", initial_gains_tensor)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """重置网络参数。"""
        linear_layers: list[nn.Linear] = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if not linear_layers:
            return

        # 输出层以稳定的已调 PID 增益开始；在线 BP 只做小幅修正。
        output_layer = linear_layers[-1]
        nn.init.zeros_(output_layer.weight)
        with torch.no_grad():
            ratio = torch.clamp(
                self.initial_gains / self.gain_limits,
                min=1e-4,
                max=1.0 - 1e-4,
            )
            output_layer.bias.copy_(torch.logit(ratio))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播,计算PID增益。"""
        raw = self.net(features)
        return torch.sigmoid(raw) * self.gain_limits.to(raw.device)


class BPPIDController:
    """用于单一连续小车作用力的自适应BP-PID控制器。

    环境观测值包括:
    ``[小车速度, 极点1角度, 极点2角度, 极点1角速度, 极点2角速度, 小车位置]``

    将双曲正切函数作为激活函数，所有值都被归一化到 ``[-1, 1]`` 范围内。
    控制器首先将多变量的平衡目标整合成一个有符号误差,然后让一个小型BP网络根据当前的PID特征生成PID增益。
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        hidden_dim: int = 64,
        gain_limits: Iterable[float] = (2.0, 0.05, 0.2),
        initial_gains: Iterable[float] = (1.0, 0.0, 0.0),
        error_weights: Iterable[float] = (
            -2.73093865,
            14.2876764,
            -19.57212661,
            1.08134621,
            0.56824623,
            1.05017658,
        ),
        integral_limit: float = 0.5,
        derivative_limit: float = 8.0,
        action_limit: float = 1.0,
        learning_rate: float = 1e-5,
        gain_regularization: float = 1e-2,
        device: str | torch.device | None = None,
    ) -> None:
        self.dt = float(dt)
        self.error_weights = np.asarray(tuple(error_weights), dtype=np.float32)
        self.integral_limit = float(integral_limit)
        self.derivative_limit = float(derivative_limit)
        self.action_limit = float(action_limit)
        self.gain_regularization = float(gain_regularization)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.initial_gains = torch.as_tensor(
            tuple(initial_gains), dtype=torch.float32, device=self.device
        )

        self.network = BPNetwork(
            input_dim=4,
            hidden_dim=hidden_dim,
            gain_limits=gain_limits,
            initial_gains=initial_gains,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.reset()

    def reset(self) -> None:
        """重置控制器状态。"""
        self.prev_error: float | None = None
        self.prev_action = 0.0
        self.integral = 0.0
        self._last_raw_action: torch.Tensor | None = None
        self._last_action_tensor: torch.Tensor | None = None
        self._last_gains: torch.Tensor | None = None
        self._last_pid_terms: torch.Tensor | None = None
        self._last_error = 0.0
        self._last_previous_action = 0.0
        self.last_snapshot: PIDSnapshot | None = None

    def observation_error(self, observation: np.ndarray) -> float:
        """根据观测值计算加权误差。"""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (6,):
            raise ValueError(f"期望的观测值形状为 (6,), 但得到 {obs.shape}")
        error = float(np.dot(self.error_weights, obs))
        if not np.isfinite(error):
            raise ValueError(f"控制误差不是有限数值: {error}")
        return error

    def act(self, observation: np.ndarray) -> tuple[np.ndarray, PIDSnapshot]:
        """根据观测值计算并执行一个动作。"""
        error = self.observation_error(observation)
        previous_error = error if self.prev_error is None else self.prev_error
        self.integral = float(
            np.clip(
                self.integral + error * self.dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative = float(
            np.clip(
                (error - previous_error) / self.dt,
                -self.derivative_limit,
                self.derivative_limit,
            )
        )

        # PID特征: [当前误差, 积分项, 微分项, 上一时刻误差]
        features = torch.tensor(
            [error, self.integral, derivative, previous_error],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # BP网络输出PID增益
        gains = self.network(features).squeeze(0)
        pid_terms = torch.tensor(
            [error, self.integral, derivative],
            dtype=torch.float32,
            device=self.device,
        )
        # 计算原始动作
        raw_action = torch.dot(gains, pid_terms)
        # 对动作进行缩放和裁剪
        action_tensor = torch.tanh(raw_action) * self.action_limit
        action = float(action_tensor.detach().cpu().item())

        # 存储用于学习的中间变量
        self._last_raw_action = raw_action
        self._last_action_tensor = action_tensor
        self._last_gains = gains
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
        """使用经典的被控对象灵敏度符号作为替代梯度,进行一次在线BP-PID更新。

        由于Pymunk是不可微分的黑盒,无法通过环境进行精确的反向传播。
        此更新方法使用最新的有限差分被控对象响应的符号（替代梯度）
        来通过PID控制律网络进行反向传播。这种方法设计上是保守的;
        在评估器中使用 ``adapt=True`` 以进行在线调整。
        """

        if self._last_raw_action is None or self._last_action_tensor is None:
            return 0.0

        # 计算下一个状态的误差 (t+1时刻的误差)
        next_error = self.observation_error(next_observation)

        # 计算动作增量 (t时刻与t-1时刻的动作差值)
        action_delta = (
            self.last_snapshot.action - self._last_previous_action
            if self.last_snapshot
            else 0.0
        )
        # 计算误差增量 (t+1时刻与t时刻的误差差值)
        error_delta = next_error - self._last_error

        # 近似计算被控对象的敏感度符号 (sgn(dy/du))
        # 这反映了上一步的动作变化对当前误差变化方向的影响
        if abs(action_delta) < 1e-6:
            plant_sign = 1.0
        else:
            plant_sign = float(np.sign(error_delta / action_delta))
            if plant_sign == 0.0:
                plant_sign = 1.0

        error_tensor = torch.tensor(next_error, dtype=torch.float32, device=self.device)
        sign_tensor = torch.tensor(plant_sign, dtype=torch.float32, device=self.device)
        # 正则化项防止在线学习把稳定初始增益推到激进区域。
        regularizer = 1e-4 * self._last_raw_action.pow(2)
        if self._last_gains is not None:
            regularizer = regularizer + self.gain_regularization * torch.mean(
                (self._last_gains - self.initial_gains) ** 2
            )

        # 核心在线学习损失函数设计：
        # 根据链式法则 dL/dW = dL/de * de/du * du/dW，由于de/du未知，我们用 plant_sign 近似。
        # 我们定义 loss = E_{t+1} * sign(de/du) * u_t。
        # 当调用 loss.backward() 时，对上一步的输出 _last_raw_action 求导，即可得到一个与符号梯度匹配的伪目标。
        loss = (
            error_tensor.detach() * sign_tensor.detach() * self._last_action_tensor
            + regularizer
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def save(self, path: str | Path) -> None:
        """保存控制器模型和优化器状态。"""
        payload = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "dt": self.dt,
            "error_weights": self.error_weights.tolist(),
            "integral_limit": self.integral_limit,
            "derivative_limit": self.derivative_limit,
            "action_limit": self.action_limit,
            "initial_gains": self.initial_gains.detach().cpu().tolist(),
            "gain_regularization": self.gain_regularization,
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        """加载控制器模型和优化器状态。"""
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
        self.derivative_limit = float(payload.get("derivative_limit", self.derivative_limit))
        self.action_limit = float(payload.get("action_limit", self.action_limit))
        self.gain_regularization = float(
            payload.get("gain_regularization", self.gain_regularization)
        )
        if "initial_gains" in payload:
            self.initial_gains = torch.as_tensor(
                payload["initial_gains"], dtype=torch.float32, device=self.device
            )
        self.reset()
