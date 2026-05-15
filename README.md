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

- 手动创建 Gymnasium 环境

```python
import gymnasium as gym
import envs.double_cartpole

env = gym.make("double-cartpole-custom-v0", n_steps=1000)
obs, info = env.reset(seed=42)
```
