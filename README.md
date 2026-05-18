- 代码组织结构

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
  *   `--learning-rate <浮点数>`: 在线学习率 (默认: 1e-5)。
  *   `--gain-limits KP_MAX KI_MAX KD_MAX`: BP网络输出的PID增益上限。
  *   `--initial-gains KP KI KD`: BP-PID的稳定初始增益。
  *   `--error-weights CV A1 A2 W1 W2 X`: 6维观测到单一控制误差的线性权重。
  *   `--pole-1-angle-deg <角度>` / `--pole-2-angle-deg <角度>`: 固定两根摆杆的初始角度，便于复现实验。

- **示例**:
  加载名为 `my_model.pt` 的模型进行10个回合的评估，关闭在线学习，并实时渲染：
  ```bash
  python -m evaluation.evaluate_bp --model my_model.pt --episodes 10 --no-adapt --render
  ```

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
