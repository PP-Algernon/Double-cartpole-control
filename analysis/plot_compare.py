"""Compare saved BP-PID experiment runs.

This script reads one or more run directories produced by
``python -m evaluation.evaluate_bp --run-dir ...`` and saves overlay plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="对比多个BP-PID实验run-dir的曲线。")
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="实验目录或 steps.csv 文件路径。",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="每个实验的图例名称；数量应与 runs 一致。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/compare"),
        help="对比图保存目录。",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="每条逐步曲线最多绘制的点数，便于快速查看长实验。",
    )
    return parser.parse_args()


def _resolve_steps_csv(path: Path) -> Path:
    """Resolve either a run directory or a direct CSV path."""
    if path.is_dir():
        path = path / "steps.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到逐步数据文件: {path}")
    return path


def _default_label(path: Path) -> str:
    """Build a readable default label."""
    if path.is_file():
        return path.parent.name or path.stem
    return path.name


def _read_step_csv(path: Path) -> dict[str, list[float]]:
    """Read numeric columns from a saved steps.csv."""
    columns: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, [])
                if value in ("True", "False"):
                    columns[key].append(float(value == "True"))
                else:
                    columns[key].append(float(value))
    return columns


def _read_summary(run_path: Path) -> dict[str, Any]:
    """Read summary.json when available."""
    summary_path = run_path / "summary.json" if run_path.is_dir() else run_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _limit(values: list[float], max_points: int | None) -> list[float]:
    """Limit plotted points without changing stored raw data."""
    if max_points is None:
        return values
    return values[:max_points]


def _plot_runs(
    runs: list[dict[str, Any]],
    output_dir: Path,
    max_points: int | None,
) -> list[Path]:
    """Create overlay plots for common diagnostics."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-double-cartpole")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

    plt.figure(figsize=(10, 5))
    for run in runs:
        data = run["data"]
        episode = np.asarray(data["episode"], dtype=np.int64)
        lengths = []
        rewards = []
        for ep in np.unique(episode):
            mask = episode == ep
            lengths.append(int(np.count_nonzero(mask)))
            rewards.append(float(np.sum(np.asarray(data["reward"], dtype=np.float64)[mask])))
        plt.plot(np.arange(len(rewards)), rewards, marker="o", label=run["label"])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Episode reward comparison")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save("compare_episode_rewards.png")

    plt.figure(figsize=(10, 5))
    for run in runs:
        data = run["data"]
        episode = np.asarray(data["episode"], dtype=np.int64)
        lengths = [int(np.count_nonzero(episode == ep)) for ep in np.unique(episode)]
        plt.plot(np.arange(len(lengths)), lengths, marker="o", label=run["label"])
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode length comparison")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    save("compare_episode_lengths.png")

    line_specs = [
        ("obs1_pole1_angle", "Pole 1 angle", "compare_pole1_angle.png"),
        ("obs2_pole2_angle", "Pole 2 angle", "compare_pole2_angle.png"),
        ("obs5_cart_distance", "Cart distance", "compare_cart_distance.png"),
        ("action", "Action", "compare_action.png"),
        ("error", "Control error", "compare_error.png"),
        ("kp", "Kp", "compare_kp.png"),
        ("ki", "Ki", "compare_ki.png"),
        ("kd", "Kd", "compare_kd.png"),
    ]
    for column, title, filename in line_specs:
        plt.figure(figsize=(10, 5))
        for run in runs:
            values = _limit(run["data"][column], max_points)
            plt.plot(np.arange(len(values)), values, label=run["label"], linewidth=1.2)
        plt.xlabel("Global step")
        plt.ylabel(column)
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        save(filename)

    return saved


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels 的数量必须与 runs 数量一致")

    runs: list[dict[str, Any]] = []
    for idx, run_path in enumerate(args.runs):
        csv_path = _resolve_steps_csv(run_path)
        runs.append(
            {
                "path": run_path,
                "csv": csv_path,
                "label": args.labels[idx] if args.labels else _default_label(run_path),
                "data": _read_step_csv(csv_path),
                "summary": _read_summary(run_path),
            }
        )

    saved = _plot_runs(runs, args.output_dir, args.max_points)
    print(f"已保存 {len(saved)} 张对比图到 {args.output_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
