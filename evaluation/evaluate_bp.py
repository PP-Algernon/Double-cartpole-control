"""用于Gymnasium双倒立摆环境的BP-PID控制器命令行评估工具。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在双倒立摆上评估BP-PID控制器。")
    parser.add_argument("--episodes", type=int, default=5, help="评估的回合数。")
    parser.add_argument("--steps", type=int, default=1000, help="每个回合的最大步数。")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子。")
    parser.add_argument(
        "--render",
        action="store_true",
        help="使用Pygame进行渲染,即Gymnasium的render_mode='human'。",
    )
    parser.add_argument(
        "--adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每次状态转移后启用在线BP更新。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="要加载的可选控制器检查点文件路径。",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="保存控制器检查点的可选路径。",
    )
    parser.add_argument(
        "--save-each-episode",
        action="store_true",
        help="每个回合结束后保存一次控制器参数；配合 --save-model 或 --checkpoint-dir 使用。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="保存逐回合检查点的目录，会写入 latest.pt 和 episode_XXXX.pt。",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="记录每一步诊断信息的可选CSV文件路径。",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="保存本次实验所有输出的目录；会自动生成 steps.csv、summary.json 和 figures/。",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="保存本次评估摘要和参数配置的JSON路径。",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="根据逐步CSV保存曲线图的目录。",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="指定Torch设备。默认为可用时使用cuda。",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="BP网络隐藏层的宽度。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="在线BP学习率。",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1200.0,
        help="环境的作用力缩放系数。",
    )
    parser.add_argument(
        "--gain-limits",
        type=float,
        nargs=3,
        metavar=("KP_MAX", "KI_MAX", "KD_MAX"),
        default=(3.0, 0.05, 0.4),
        help="BP网络输出的PID增益上限。",
    )
    parser.add_argument(
        "--initial-gains",
        type=float,
        nargs=3,
        metavar=("KP", "KI", "KD"),
        default=(1.0, 0.0, 0.0),
        help="BP-PID的稳定初始增益。",
    )
    parser.add_argument(
        "--error-weights",
        type=float,
        nargs=6,
        metavar=("CV", "A1", "A2", "W1", "W2", "X"),
        default=None,
        help="6维观测到单一控制误差的线性权重。",
    )
    parser.add_argument(
        "--integral-limit",
        type=float,
        default=0.35,
        help="PID积分项限幅。",
    )
    parser.add_argument(
        "--derivative-limit",
        type=float,
        default=6.0,
        help="PID微分项限幅。",
    )
    parser.add_argument(
        "--pole-1-angle-deg",
        type=float,
        default=None,
        help="第一根摆杆的固定初始角度,单位为度。",
    )
    parser.add_argument(
        "--pole-2-angle-deg",
        type=float,
        default=None,
        help="第二根摆杆的固定初始角度,单位为度。",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    """将命令行参数转换为可JSON序列化的值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _resolve_output_paths(args: argparse.Namespace) -> None:
    """根据 run-dir 补齐CSV、JSON和图像输出路径。"""
    if args.run_dir is None:
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.csv is None:
        args.csv = args.run_dir / "steps.csv"
    if args.summary_json is None:
        args.summary_json = args.run_dir / "summary.json"
    if args.plot_dir is None:
        args.plot_dir = args.run_dir / "figures"


def _csv_writer(path: Path | None) -> tuple[Any | None, csv.DictWriter | None]:
    """初始化CSV写入器。"""
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "episode",
        "step",
        "reward",
        "terminated",
        "truncated",
        "loss",
        "force",
        "obs0_cart_velocity",
        "obs1_pole1_angle",
        "obs2_pole2_angle",
        "obs3_pole1_angular_velocity",
        "obs4_pole2_angular_velocity",
        "obs5_cart_distance",
        "error",
        "integral",
        "derivative",
        "kp",
        "ki",
        "kd",
        "action",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def _read_step_csv(path: Path) -> dict[str, list[float]]:
    """读取逐步诊断CSV，返回列数据。"""
    columns: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if key not in columns:
                    columns[key] = []
                if value in ("True", "False"):
                    columns[key].append(float(value == "True"))
                else:
                    columns[key].append(float(value))
    return columns


def _plot_step_csv(csv_path: Path, plot_dir: Path) -> list[Path]:
    """根据逐步CSV生成常用诊断曲线。"""
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-double-cartpole")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir.mkdir(parents=True, exist_ok=True)
    data = _read_step_csv(csv_path)
    if not data:
        return []

    episode = np.asarray(data["episode"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.float64)
    global_step = np.arange(step.size, dtype=np.float64)
    saved: list[Path] = []

    def save_current(name: str) -> None:
        path = plot_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

    plt.figure(figsize=(10, 5))
    for ep in np.unique(episode):
        mask = episode == ep
        rewards = np.asarray(data["reward"], dtype=np.float64)[mask]
        plt.plot(step[mask], np.cumsum(rewards), label=f"episode {ep}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.title("Episode cumulative reward")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("reward_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["obs1_pole1_angle"], label="pole1 angle")
    plt.plot(global_step, data["obs2_pole2_angle"], label="pole2 angle")
    plt.axhline(1.0, color="r", linestyle="--", linewidth=1)
    plt.axhline(-1.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Global step")
    plt.ylabel("Normalized angle")
    plt.title("Pole angles")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("pole_angles.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["obs3_pole1_angular_velocity"], label="pole1 angular velocity")
    plt.plot(global_step, data["obs4_pole2_angular_velocity"], label="pole2 angular velocity")
    plt.xlabel("Global step")
    plt.ylabel("Normalized angular velocity")
    plt.title("Pole angular velocities")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("pole_angular_velocities.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["obs5_cart_distance"], label="cart distance")
    plt.plot(global_step, data["obs0_cart_velocity"], label="cart velocity")
    plt.axhline(1.0, color="r", linestyle="--", linewidth=1)
    plt.axhline(-1.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Global step")
    plt.ylabel("Normalized value")
    plt.title("Cart state")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("cart_state.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["action"], label="action")
    plt.plot(global_step, np.asarray(data["force"], dtype=np.float64) / 1200.0, label="force / 1200")
    plt.axhline(1.0, color="r", linestyle="--", linewidth=1)
    plt.axhline(-1.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Global step")
    plt.ylabel("Command")
    plt.title("Control command")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("control_command.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["kp"], label="Kp")
    plt.plot(global_step, data["ki"], label="Ki")
    plt.plot(global_step, data["kd"], label="Kd")
    plt.xlabel("Global step")
    plt.ylabel("Gain")
    plt.title("BP-PID gains")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("pid_gains.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["error"], label="error")
    plt.plot(global_step, data["integral"], label="integral")
    plt.plot(global_step, data["derivative"], label="derivative")
    plt.xlabel("Global step")
    plt.ylabel("PID term")
    plt.title("PID terms")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("pid_terms.png")

    plt.figure(figsize=(10, 5))
    plt.plot(global_step, data["loss"], label="loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Online BP pseudo-loss")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save_current("bp_loss.png")

    return saved


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    """评估主函数。"""
    import numpy as np
    import torch

    from controllers.bp_pid import BPPIDController
    from envs.double_cartpole import (
        DOUBLE_CARTPOLE_ENV_ID,
        make_double_cartpole,
        register_double_cartpole,
    )

    _resolve_output_paths(args)
    register_double_cartpole()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    render_mode = "human" if args.render else None
    env = make_double_cartpole(
        render_mode=render_mode,
        n_steps=args.steps,
        force_scale=args.force_scale,
    )
    controller_kwargs = {
        "dt": getattr(env.unwrapped, "dt", 1.0 / 60.0),
        "hidden_dim": args.hidden_dim,
        "gain_limits": args.gain_limits,
        "initial_gains": args.initial_gains,
        "integral_limit": args.integral_limit,
        "derivative_limit": args.derivative_limit,
        "learning_rate": args.learning_rate,
        "device": args.device,
    }
    if args.error_weights is not None:
        controller_kwargs["error_weights"] = args.error_weights
    controller = BPPIDController(**controller_kwargs)
    if args.model is not None:
        controller.load(args.model, load_optimizer=args.adapt)

    csv_handle, writer = _csv_writer(args.csv)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    termination_count = 0
    truncation_count = 0

    try:
        for episode in range(args.episodes):
            reset_options = {}
            if args.pole_1_angle_deg is not None:
                reset_options["pole_1_angle"] = np.deg2rad(args.pole_1_angle_deg)
            if args.pole_2_angle_deg is not None:
                reset_options["pole_2_angle"] = np.deg2rad(args.pole_2_angle_deg)
            observation, _ = env.reset(
                seed=args.seed + episode,
                options=reset_options or None,
            )
            controller.reset()
            total_reward = 0.0
            final_step = 0

            for step in range(args.steps):
                # 1. 控制器前向传播:计算误差，通过BP网络输出自适应PID增益，并计算当前动作。
                action, snapshot = controller.act(observation)

                # 2. 环境步进:在不可微分的实际环境(gym)中执行动作。
                next_observation, reward, terminated, truncated, info = env.step(action)

                # 3. 在线BP学习 (如果 args.adapt=True):
                #    通过评估新误差(next_observation)，近似被控对象灵敏度(dy/du，使用符号差异因子)，
                #    并更新神经网络参数以匹配梯度方向，实现在线学习。
                loss = controller.learn_from_transition(next_observation) if args.adapt else 0.0

                total_reward += float(reward)
                final_step = step + 1

                if writer is not None:
                    row = {
                        "episode": episode,
                        "step": step,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "loss": float(loss),
                        "force": float(info.get("force", 0.0)),
                    }
                    row.update(
                        {
                            f"obs{idx}_{name}": float(value)
                            for idx, (name, value) in enumerate(
                                zip(
                                    [
                                        "cart_velocity",
                                        "pole1_angle",
                                        "pole2_angle",
                                        "pole1_angular_velocity",
                                        "pole2_angular_velocity",
                                        "cart_distance",
                                    ],
                                    next_observation,
                                )
                            )
                        }
                    )
                    row.update(asdict(snapshot))
                    writer.writerow(row)

                observation = next_observation
                if terminated or truncated:
                    termination_count += int(terminated)
                    truncation_count += int(truncated)
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(final_step)

            if args.save_each_episode or args.checkpoint_dir is not None:
                checkpoint_path = args.save_model
                if args.checkpoint_dir is not None:
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    episode_path = args.checkpoint_dir / f"episode_{episode + 1:04d}.pt"
                    controller.save(episode_path)
                    checkpoint_path = args.checkpoint_dir / "latest.pt"
                if checkpoint_path is None:
                    raise ValueError(
                        "使用 --save-each-episode 时需要同时指定 --save-model 或 --checkpoint-dir"
                    )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                controller.save(checkpoint_path)

            print(
                f"episodes={episode + 1}/{args.episodes} "
                f"reward={total_reward:.2f} length={final_step}"
            )
    finally:
        env.close()
        if csv_handle is not None:
            csv_handle.close()

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        controller.save(args.save_model)

    rewards = np.asarray(episode_rewards, dtype=np.float64)
    lengths = np.asarray(episode_lengths, dtype=np.float64)
    summary = {
        "episodes": float(args.episodes),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "std_reward": float(rewards.std(ddof=0)) if rewards.size else 0.0,
        "mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "terminations": float(termination_count),
        "truncations": float(truncation_count),
    }

    saved_figures: list[Path] = []
    if args.plot_dir is not None and args.csv is not None:
        saved_figures = _plot_step_csv(args.csv, args.plot_dir)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "terminations": termination_count,
            "truncations": truncation_count,
            "args": {
                key: _json_safe(value)
                for key, value in vars(args).items()
            },
            "outputs": {
                "csv": str(args.csv) if args.csv is not None else None,
                "figures": [str(path) for path in saved_figures],
                "save_model": str(args.save_model) if args.save_model is not None else None,
                "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir is not None else None,
            },
            "controller": {
                "device": str(controller.device),
                "torch": torch.__version__,
                "error_weights": controller.error_weights.tolist(),
                "integral_limit": controller.integral_limit,
                "derivative_limit": controller.derivative_limit,
                "action_limit": controller.action_limit,
                "initial_gains": controller.initial_gains.detach().cpu().tolist(),
                "gain_regularization": controller.gain_regularization,
            },
        }
        args.summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        "评估摘要\n "
        f"环境={DOUBLE_CARTPOLE_ENV_ID}\n "
        f"平均奖励={summary['mean_reward']:.2f}\n "
        f"奖励标准差={summary['std_reward']:.2f}\n "
        f"平均步长={summary['mean_length']:.1f}\n "
        f"终止次数={termination_count}\n "
        f"截断次数={truncation_count}\n"
    )
    if args.csv is not None:
        print(f"逐步数据={args.csv}")
    if args.summary_json is not None:
        print(f"实验摘要={args.summary_json}")
    if saved_figures:
        print(f"图像目录={args.plot_dir}")
    return summary


def main() -> None:
    """主入口函数。"""
    evaluate(parse_args())


if __name__ == "__main__":
    main()
