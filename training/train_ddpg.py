"""Train a DDPG policy on the project double-cartpole environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.double_cartpole import make_double_cartpole


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--steps", type=int, default=1000, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-scale", type=float, default=1200.0)
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs/ddpg"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/models/ddpg"))
    parser.add_argument("--save-name", type=str, default="ddpg_double_cartpole")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--resume", type=Path, default=None, help="path to an existing DDPG zip")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-noise-sigma", type=float, default=0.15)
    return parser.parse_args()


def make_vec_env(*, seed: int, n_steps: int, force_scale: float, log_dir: Path) -> DummyVecEnv:
    log_dir.mkdir(parents=True, exist_ok=True)

    def _factory():
        env = make_double_cartpole(n_steps=n_steps, force_scale=force_scale)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return DummyVecEnv([_factory])


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        seed=args.seed,
        n_steps=args.steps,
        force_scale=args.force_scale,
        log_dir=args.log_dir / "train",
    )
    eval_env = make_vec_env(
        seed=args.seed + 10_000,
        n_steps=args.steps,
        force_scale=args.force_scale,
        log_dir=args.log_dir / "eval",
    )

    if args.resume is not None:
        model = DDPG.load(args.resume, env=train_env, device=args.device)
    else:
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.action_noise_sigma * np.ones(n_actions),
        )
        model = DDPG(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=1,
            gradient_steps=1,
            action_noise=action_noise,
            tensorboard_log=str(args.log_dir / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_dir / "best"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.model_dir / "checkpoints"),
        name_prefix=args.save_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        final_path = args.model_dir / f"{args.save_name}_final"
        model.save(final_path)
        print(f"saved final DDPG model to {final_path}.zip")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
