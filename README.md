- 代码组织结构

minimind仓库：https://github.com/jingyaogong/minimind
视频github仓库：https://github.com/Wood-Q/MokioMind
视频notion笔记：https://mirage-thought-d06.notion.site/minimind-3-29c747825dae80bda611dd4dfe4f0e7b?pvs=74
sk-161ca43c9246415a890b1104b81202cc

为保证实验可复现和代码可维护，建议按以下模块化结构组织项目：
```
~/
├── README.md
├── requirements.txt              # 依赖：numpy, gymnasium, stable-baselines3, torch cu128, matplotlib
├── envs/
│   └── double_cartpole.py        # 环境包装器（可选，统一接口）
├── controllers/
│   ├── bp_pid.py                 # BP-PID 控制器实现
│   │   ├── class BPNetwork       # 网络结构、前向传播、反向传播
│   │   └── class BPPIDController # 控制器封装
│   └── rl_controller.py          # RL 控制器统一接口
├── training/
│   ├── train_ppo.py              # PPO 训练脚本
│   ├── train_sac.py              # SAC 训练脚本
│   └── train_ddpg.py             # DDPG 训练脚本
├── evaluation/
│   ├── evaluate_bp.py            # BP-PID 评估
│   ├── evaluate_rl.py            # RL 评估
│   ├── disturbance_test.py       # 扰动测试
│   └── robustness_test.py        # 鲁棒性测试（参数摄动）
├── analysis/
│   ├── metrics.py                # 性能指标计算
│   ├── plot_compare.py           # 对比绘图
│   └── statistical_test.py       # 统计检验（t 检验等）
└── results/
    ├── figures/                  # 结果图表
    └── logs/                     # 训练日志
```

- 推荐环境

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

- 运行 BP-PID 评估

```bash
python -m evaluation.evaluate_bp --episodes 5 --steps 1000
```



该脚本用于评估 BP-PID 控制器在双倒立摆环境中的性能。你可以通过附加命令行参数来控制评估过程。

- **开启实时渲染**:
  在命令后添加 `--render` 标志可以打开一个 Pygame 窗口，实时观看仿真过程。
  ```bash
  python -m evaluation.evaluate_bp --episodes 5 --render
  ```

- **常用参数**:
  *   `--episodes <整数>`: 运行评估的回合总数 (默认: 5)。
  *   `--steps <整数>`: 每个回合的最大步数 (默认: 1000)。
  *   `--seed <整数>`: 随机种子，用于复现 (默认: 42)。
  *   `--adapt` / `--no-adapt`: 启用或禁用在线自适应学习 (默认: 启用)。
  *   `--model <路径>`: 加载一个预训练好的 `.pt` 模型文件进行评估。
  *   `--save-model <路径>`: 将评估后的模型参数保存到指定路径。
  *   `--save-each-episode`: 每个回合结束后都保存一次最新控制器参数。
  *   `--checkpoint-dir <目录>`: 保存 `latest.pt` 和逐回合 `episode_XXXX.pt` 检查点。
  *   `--csv <路径>`: 将详细的逐帧数据（观测、动作、PID增益等）导出到 CSV 文件。
  *   `--run-dir <目录>`: 保存完整实验输出；自动写入 `steps.csv`、`summary.json` 和 `figures/`。
  *   `--summary-json <路径>`: 单独指定实验摘要 JSON 的保存位置。
  *   `--plot-dir <目录>`: 单独指定自动曲线图保存目录。
  *   `--learning-rate <浮点数>`: 在线学习率 (默认: 5e-6)。
  *   `--gain-limits KP_MAX KI_MAX KD_MAX`: BP网络输出的PID增益上限。
  *   `--initial-gains KP KI KD`: BP-PID的稳定初始增益。
  *   `--error-weights CV A1 A2 W1 W2 X`: 6维观测到单一控制误差的线性权重。
  *   `--pole-1-angle-deg <角度>` / `--pole-2-angle-deg <角度>`: 固定两根摆杆的初始角度，便于复现实验。

- **示例**:
  加载名为 `my_model.pt` 的模型进行10个回合的评估，关闭在线学习，并实时渲染：
  ```bash
  python -m evaluation.evaluate_bp --model my_model.pt --episodes 10 --no-adapt --render
  ```

- **保存完整实验数据和图像**:
  推荐每次正式实验都使用独立的 `--run-dir`，这样原始数据、摘要和曲线会放在同一个目录中：
  ```bash
  python -m evaluation.evaluate_bp --episodes 1 --steps 10000 --run-dir results/logs/bp_pid_run_001
  ```

  该目录会生成：
  *   `steps.csv`: 每一步的原始数据，包括 `episode`、`step`、`reward`、终止标志、`loss`、`force`、6维观测、PID误差项、`kp/ki/kd` 和 `action`。
  *   `summary.json`: 实验摘要、每回合奖励/步长、命令行参数、控制器配置和输出文件索引。
  *   `figures/`: 自动导出的诊断图，包括奖励曲线、摆杆角度、角速度、小车状态、控制动作、PID增益、PID项和BP伪损失。

  如果你只想指定某一类输出，也可以单独使用：
  ```bash
  python -m evaluation.evaluate_bp --episodes 10 --steps 10000 --csv results/logs/test_steps.csv --plot-dir results/figures/test
  python -m evaluation.evaluate_bp --episodes 10 --steps 10000 --summary-json results/logs/test_summary.json
  ```

- **对比多次实验曲线**:
  多个 `run-dir` 保存后，可以叠图对比：
  ```bash
  python -m analysis.plot_compare results/logs/bp_pid_run_001 results/logs/bp_pid_run_002 --labels baseline tuned --output-dir results/figures/bp_compare
  ```
  输出会包含每回合奖励、每回合步长、摆杆角度、小车位置、动作、误差和 `Kp/Ki/Kd` 的对比图。

- 手动创建 Gymnasium 环境

```python
import gymnasium as gym
import envs.double_cartpole

env = gym.make("double-cartpole-custom-v0", n_steps=1000)
obs, info = env.reset(seed=42)
```

## BP-PID 在线学习机制 (Online Learning Mechanism)

本项目在 BP-PID 控制器中引入了一种针对“不可微（黑盒）环境”的特殊在线学习（Online Tuning）机制。由于物理仿真引擎（基于 Pymunk）不可直接求导，传统的误差反向传播无法直接计算“误差对输出动作的梯度”（即植物敏感度）。项目通过采用**近似偏导数（符号梯度，Surrogate Gradient）**绕过了这一限制。

**核心原理步骤**：

1. **环境敏感度符号近似 (Plant Sensitivity Sign)**:
   利用有限差分（Finite Difference）来估计环境反馈方向。即计算物理误差增量和前一步动作增量的比值符号：
   `plant_sign = sgn(ΔError / ΔAction) = sgn((e_{t+1} - e_t) / (u_t - u_{t-1}))`
   这反映了“增大小车推力会在下一步让误差变大还是变小”。

2. **核心伪损失函数设计 (Pseudo-Loss Function)**:
   为了让 PyTorch 能够通过网络参数直接优化环境误差，代码中巧妙构建了这样一个目标函数：
   `loss = E_{t+1} * plant_sign * u_t + regularizer`

3. **链式法则驱动网络更新 (Backpropagation)**:
   由于神经网络的输出是 `u_t`，当对 `loss` 进行反向传播 `loss.backward()` 时，其相对 `u_t` 的梯度恰好等同于误差通过敏感度符号带来的方向指导（即 `d(Loss)/d(u_t) ∝ E_{t+1} * plant_sign`）。这样就能跨越环境这个不可导的黑盒，引导生成 PID 增益系数的 BP 神经网络朝着“缩小整体误差”的方向进行梯度下降。最终实现了边交互、边学习的网络自适应调节。

## BP-PID 调参思路

当前控制器分成两层：固定的 `error_weights` 先把 6 维观测压成一个有符号误差，BP 网络再根据误差、积分和微分项输出 `Kp/Ki/Kd`。因此调参时不要一开始就开在线 BP 乱试，推荐按下面顺序推进。

1. 先关掉在线学习，只调线性误差方向：
   ```bash
   python -m evaluation.evaluate_bp --episodes 5 --steps 10000 --no-adapt
   ```
   目标是让固定控制律至少稳定跑满目标步数。此时如果失败，优先看 CSV 中最后一行：若 `obs1/obs2` 先到 `±1`，说明杆角控制或角速度阻尼不足；若 `obs5` 先到 `±1`，说明小车回中项不足或推力太激进。

2. `error_weights` 的符号关系比小数更重要。当前比较稳的结构是：
   - 小车速度：负权重，用来抑制小车速度。
   - 第一根杆角度：正权重。
   - 第二根杆角度：负权重，和第一根杆反号。
   - 第一根杆角速度：小负权重，用于阻尼。
   - 第二根杆角速度：较大正权重，这是长期稳定的关键阻尼项。
   - 小车位置：负权重，让小车回到中心附近。

3. 固定误差权重稳定后，再打开在线 BP：
   ```bash
   python -m evaluation.evaluate_bp --episodes 5 --steps 10000
   ```
   如果开 BP 后变差，先降低 `--learning-rate`，再增大 `gain_regularization` 的代码默认值；不要先扩大 `gain-limits`。BP 在这里更适合做小幅 PID 增益微调，而不是救一个错误的误差定义。

4. 保存可复现实验：
   ```bash
   python -m evaluation.evaluate_bp --episodes 20 --steps 10000 --run-dir results/logs/bp_pid_run_001 --checkpoint-dir results/logs/bp_pid_checkpoints
   ```
   需要继续训练时加载 `results/logs/bp_pid_checkpoints/latest.pt`。

本次默认参数不是手动凭感觉逐个数字试出来的，而是按黑盒控制参数搜索得到的：先关闭在线 BP（`--no-adapt`），只优化 `error_weights` 对系统闭环方向的影响；再在保留符号结构的前提下做随机扰动搜索。粗筛阶段用少量种子和较短步数淘汰会快速发散的组合，复评阶段用更多种子和 `10000` 步验证长期稳定性。最终保留的结构是第二根杆角速度阻尼、小车回中项和两根杆角度的相对符号共同起作用，而不是单纯把某一个角度权重调大。
