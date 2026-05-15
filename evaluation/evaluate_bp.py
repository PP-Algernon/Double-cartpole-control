"""Command-line BP-PID evaluation for the Gymnasium double-cartpole env."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BP-PID on double cartpole.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with pygame using Gymnasium render_mode='human'.",
    )
    parser.add_argument(
        "--adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable online BP updates after each transition.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional controller checkpoint to load.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to save controller checkpoint.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional per-step diagnostics CSV path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Torch device. Default uses cuda when available.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="BP hidden layer width.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Online BP learning rate.",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1200.0,
        help="Environment force scale.",
    )
    return parser.parse_args()


def _csv_writer(path: Path | None) -> tuple[Any | None, csv.DictWriter | None]:
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


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    import numpy as np
    import torch

    from controllers.bp_pid import BPPIDController
    from envs.double_cartpole import (
        DOUBLE_CARTPOLE_ENV_ID,
        make_double_cartpole,
        register_double_cartpole,
    )

    register_double_cartpole()
    render_mode = "human" if args.render else None
    env = make_double_cartpole(
        render_mode=render_mode,
        n_steps=args.steps,
        force_scale=args.force_scale,
    )
    controller = BPPIDController(
        dt=getattr(env.unwrapped, "dt", 1.0 / 60.0),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    if args.model is not None:
        controller.load(args.model, load_optimizer=args.adapt)

    csv_handle, writer = _csv_writer(args.csv)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    termination_count = 0
    truncation_count = 0

    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            controller.reset()
            total_reward = 0.0
            final_step = 0

            for step in range(args.steps):
                action, snapshot = controller.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
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
            print(
                f"episode={episode + 1}/{args.episodes} "
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
    print(
        "summary "
        f"env={DOUBLE_CARTPOLE_ENV_ID} "
        f"device={controller.device} "
        f"torch={torch.__version__} "
        f"mean_reward={summary['mean_reward']:.2f} "
        f"std_reward={summary['std_reward']:.2f} "
        f"mean_length={summary['mean_length']:.1f} "
        f"terminations={termination_count} "
        f"truncations={truncation_count}"
    )
    return summary


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
