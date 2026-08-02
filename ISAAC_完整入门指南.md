# Isaac Sim / Isaac Lab 完整入门指南

> 本文档基于本仓库 `sim_stack` 的实际部署经验（`QUICKSTART.md`、`ISAAC_SIM_PI05_SETUP.md`）与 Isaac Lab 官方文档整理，覆盖环境部署、机器人强化学习（RL）训练、VLA（Vision-Language-Action，如 pi0.5/openpi）模型部署与联调三大主题。适合零基础入门。

## 目录

1. [核心概念与架构](#1-核心概念与架构)
2. [环境准备与安装](#2-环境准备与安装)
3. [Isaac Sim / Isaac Lab 基础使用](#3-isaac-sim--isaac-lab-基础使用)
4. [机器人强化学习训练](#4-机器人强化学习训练)
5. [VLA 模型部署（openpi / pi0.5）](#5-vla-模型部署openpi--pi05)
6. [Isaac Lab 与 VLA 联调（仿真闭环评测）](#6-isaac-lab-与-vla-联调仿真闭环评测)
7. [常见问题排查](#7-常见问题排查)
8. [参考资料](#8-参考资料)

---

## 1. 核心概念与架构

在开始之前，先理解几个核心概念：

- **Isaac Sim**：NVIDIA 出品的基于 Omniverse/PhysX 的机器人仿真器，负责物理仿真、渲染、传感器（相机、力矩传感器等）模拟。可以理解为"游戏引擎 + 物理引擎 + 机器人建模工具"。
- **Isaac Lab**：构建在 Isaac Sim 之上的**机器人学习框架**（原名 Isaac Orbit），提供类 Gym 的强化学习/模仿学习环境接口、预置机器人资产（Franka、Unitree Go2、ANYmal、人形机器人等）、以及对接 rsl_rl / rl_games / skrl / stable-baselines3 等主流 RL 库的训练脚本。
- **VLA 模型（Vision-Language-Action）**：接收图像 + 语言指令，直接输出机器人动作序列的端到端策略模型，例如 Physical Intelligence 的 **pi0 / pi0-FAST / pi0.5**（开源实现为 `openpi`）。
- **本仓库的整体架构**：由于 Isaac Sim 与 openpi 对 Python/依赖版本要求冲突（尤其是 numpy 版本），实践中采用**双进程 + WebSocket 通信**的方案，而不是把两者装进同一个环境：

```
┌─────────────────────────┐        WebSocket         ┌──────────────────────────┐
│  isaac_sim_env (py3.12)  │ ───── 图像+状态 ────────▶ │  openpi_env (py3.11)     │
│  Isaac Sim + Isaac Lab   │                           │  openpi / pi0.5 策略服务  │
│  bridge/isaac_lab_pi05_  │ ◀──── 动作序列 ─────────  │  scripts/serve_policy.py │
│  eval.py (仿真+桥接脚本) │                           │  (独立 GPU 运行)          │
└─────────────────────────┘                           └──────────────────────────┘
```

推荐使用**两块 GPU**：一块跑 Isaac Sim 渲染/物理，一块跑 VLA 推理，通过 `CUDA_VISIBLE_DEVICES` 隔离，避免显存冲突（JAX 默认会预占约 75% 显存，容易导致 `CUDA illegal memory access`）。

---

## 2. 环境准备与安装

### 2.1 系统要求

| 项目      | 要求                                                                                            |
| --------- | ----------------------------------------------------------------------------------------------- |
| 操作系统  | Ubuntu 22.04（openpi 强制要求，Isaac Sim 官方推荐）                                             |
| GPU       | NVIDIA GPU，建议 RTX 40/50 系列或 A100/H100；显存 ≥ 8GB（仅推理）；本仓库实测环境为双 RTX 5080 |
| 驱动/CUDA | 较新驱动（实测 580.x，CUDA 13.0），需支持你所用 GPU 架构（如 Blackwell sm_120）                 |
| Python    | Isaac Sim/Isaac Lab 用 3.12（或 3.10/3.11，视版本而定）；openpi 用 3.11                         |
| 包管理    | 推荐使用[`uv`](https://github.com/astral-sh/uv)（比 pip/conda 快很多，本仓库全程使用）         |

> Isaac Lab 官方 pip 安装文档要求：Isaac Sim 5.x 对应 Python 3.11，Isaac Sim 4.5 对应 Python 3.10。本仓库使用 Isaac Sim 6.0.1 + Python 3.12，请以你实际安装的版本文档为准。

### 2.2 安装 Isaac Sim（pip 方式）

```bash
mkdir -p ~/sim_stack && cd ~/sim_stack

# 创建独立虚拟环境
uv venv isaac_sim_env --python 3.12
source isaac_sim_env/bin/activate

# 安装 Isaac Sim（版本号按需替换）
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease allow

# 接受 EULA 并验证安装
OMNI_KIT_ACCEPT_EULA=YES python -c "import isaacsim"

deactivate
```

也可以用官方通用命令（不锁定版本）：

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# 安装匹配的 PyTorch（x86_64 示例）
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

安装完成后可直接运行 `isaacsim` 命令验证图形界面能否启动。

### 2.3 安装 Isaac Lab

```bash
cd ~/sim_stack
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout release/3.0.0-beta2   # 或对应你 Isaac Sim 版本的分支/tag

# 需要 cmake / build-essential（robomimic 依赖，Linux）
sudo apt install cmake build-essential

# 在 isaac_sim_env 中安装（会自动装好 rl_games/rsl_rl/sb3/skrl/robomimic 等）
source ~/sim_stack/isaac_sim_env/bin/activate
./isaaclab.sh --install
deactivate
```

只想装某个 RL 库时可以指定：

```bash
./isaaclab.sh -i rsl_rl   # 仅安装 rsl_rl
```

**已知坑（beta 分支资产路径 bug）**：如果 Franka 机械臂资产加载报路径错误，执行：

```bash
sed -i 's#Robots/FrankaEmika/panda_instanceable.usd#Robots/FrankaEmika/Legacy/panda_instanceable.usd#' \
    ~/sim_stack/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py
```

安装验证（应能弹出一个空场景窗口）：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

### 2.4 安装 openpi（VLA 模型库，独立环境）

```bash
cd ~/sim_stack
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi
cd openpi

# 用 uv 创建独立于 Isaac 的 Python 3.11 环境
UV_PROJECT_ENVIRONMENT=~/sim_stack/openpi_env GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

再把 `openpi-client`（轻量 WebSocket 客户端，供 Isaac 侧调用）装进 `isaac_sim_env`，**注意用 `--no-deps`** 避免 numpy 版本冲突（openpi-client 要 numpy<2.0，Isaac Sim 要 numpy≥2.x）：

```bash
source ~/sim_stack/isaac_sim_env/bin/activate
uv pip install --no-deps -e ~/sim_stack/openpi/packages/openpi-client
deactivate
```

---

## 3. Isaac Sim / Isaac Lab 基础使用

### 3.1 目录结构速览（Isaac Lab）

```
IsaacLab/
├── source/
│   ├── isaaclab/            # 核心框架：场景、传感器、控制器、环境基类
│   ├── isaaclab_assets/     # 预置机器人/物体 USD 资产配置
│   ├── isaaclab_tasks/      # 各类任务环境定义（Isaac-Ant-v0 等注册在这里）
│   ├── isaaclab_rl/         # RL 库适配层
│   ├── isaaclab_mimic/      # 模仿学习数据生成
│   └── isaaclab_teleop/     # 遥操作接口
├── scripts/
│   ├── tutorials/           # 官方教程脚本（入门必看）
│   ├── environments/        # 环境随机操作演示（random_agent 等）
│   ├── demos/               # 多机器人/多任务演示
│   ├── reinforcement_learning/   # RL 训练脚本（rsl_rl/rl_games/skrl/sb3/ray）
│   ├── imitation_learning/  # 模仿学习/数据回放
│   └── sim2sim_transfer/    # 仿真到仿真迁移
└── isaaclab.sh              # 统一命令行入口（等价于用内置 python 执行脚本）
```

### 3.2 `isaaclab.sh` 常用命令

```bash
./isaaclab.sh -p <script.py> [args]   # 用 Isaac Lab 内置 Python 运行脚本
./isaaclab.sh -i [rl_library]         # 安装依赖（可指定具体 RL 库）
./isaaclab.sh -p -m tensorboard.main --logdir=logs   # 启动 TensorBoard 看训练曲线
```

### 3.3 快速跑一个示例任务（随机动作）

```bash
source ~/sim_stack/isaac_sim_env/bin/activate
cd ~/sim_stack/IsaacLab
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Ant-v0 --num_envs 32
```

任务名格式一般为 `Isaac-<任务>-<机器人>-v0`，例如 `Isaac-Reach-Franka-v0`、`Isaac-Velocity-Flat-Unitree-Go2-v0`、`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`。全部任务在 `source/isaaclab_tasks` 中以 gym 注册的形式定义，可以 `grep -r "gym.register" source/isaaclab_tasks` 查找。

---

## 4. 机器人强化学习训练

Isaac Lab 支持 4 大主流 RL 库：**rsl_rl**（腿式机器人常用，NVIDIA 自研，速度快）、**rl_games**、**skrl**（支持多智能体 MAPPO/IPPO，及 JAX 后端）、**stable-baselines3**。所有脚本位于 `scripts/reinforcement_learning/<库名>/`。

### 4.1 通用训练/评测流程（以 rsl_rl 为例）

```bash
source ~/sim_stack/isaac_sim_env/bin/activate
cd ~/sim_stack/IsaacLab

# 训练（--headless 表示不开图形界面，速度更快，适合服务器）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless --num_envs 4096 --max_iterations 300

# 训练日志/权重默认存放在：
# logs/rsl_rl/<任务名>/<时间戳>/model_<迭代数>.pt

# 评测/回放训练好的策略（可视化 + 录像）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/unitree_go2_flat/<时间戳>/model_<迭代数>.pt \
    --video --video_length 200
```

其他库用法完全对称，只需替换库名与脚本路径：

```bash
# rl_games
./isaaclab.sh -i rl_games
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Ant-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint <ckpt>

# skrl（支持 JAX 后端 --ml_framework jax；多智能体加 --algorithm MAPPO）
./isaaclab.sh -i skrl
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-Franka-v0 --headless

# stable-baselines3
./isaaclab.sh -i sb3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
```

### 4.2 常用参数

| 参数                            | 说明                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| `--headless`                  | 无图形界面运行，训练时强烈推荐（大幅提速）                       |
| `--num_envs`                  | 并行环境数（训练常用几千个并行环境提高吞吐，评测时调小方便观察） |
| `--checkpoint`                | 指定加载的模型权重路径                                           |
| `--use_pretrained_checkpoint` | 使用官方预训练权重，无需指定路径                                 |
| `--video --video_length N`    | 评测时录制前 N 步视频（需系统装有`ffmpeg`）                    |
| `--max_iterations`            | 训练总迭代次数                                                   |
| `--seed`                      | 随机种子                                                         |

### 4.3 查看训练曲线

```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

浏览器打开提示的地址（通常是 `http://localhost:6006`）即可看 reward、loss 等曲线。

### 4.4 调参入口

每个任务的超参数（如 PPO 的学习率、网络结构）定义在对应任务目录下的 `agents/<rl_library>_ppo_cfg.py` 中（例如 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py`），直接修改该文件或用命令行覆盖对应字段。

---

## 5. VLA 模型部署（openpi / pi0.5）

### 5.1 什么是 pi0 / pi0.5

pi0 / pi0-FAST / pi0.5 是 Physical Intelligence 开源的 VLA（视觉-语言-动作）基础模型，输入是相机图像 + 机器人本体状态 + 自然语言指令，输出未来若干步的动作序列（chunk）。`openpi` 是其官方开源实现。

### 5.2 启动策略推理服务

策略以 WebSocket 服务的形式独立运行（推荐单独一块 GPU）：

```bash
source ~/sim_stack/openpi_env/bin/activate
cd ~/sim_stack/openpi

# --env 指定预置的评测环境配置（如 LIBERO），--port 为监听端口
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python scripts/serve_policy.py --env LIBERO --port 31437
```

首次运行会自动从 `~/.cache/openpi/` 下载模型权重（pi0.5 权重约 11.6GB）。

加载自定义微调权重：

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config名称> \
    --policy.dir=<checkpoint目录>
```

`XLA_PYTHON_CLIENT_MEM_FRACTION` 用于限制 JAX 预占显存比例（JAX 默认几乎占满显存，与 Isaac Sim 共卡运行时必须限制，否则容易 `CUDA illegal memory access`）。

### 5.3 GPU 显存需求参考

| 用途      | 最低显存 | 推荐 GPU        |
| --------- | -------- | --------------- |
| 仅推理    | > 8GB    | RTX 4090        |
| LoRA 微调 | > 22.5GB | RTX 4090 / A100 |
| 全量微调  | > 70GB   | A100 / H100     |

### 5.4 微调自己的数据（简述）

1. 将数据转换为 LeRobot 格式。
2. 编写对应的 `Inputs` / `Outputs` / `DataConfig` 类（定义观测/动作维度映射）。
3. 计算归一化统计量：
   ```bash
   uv run scripts/compute_norm_stats.py --config-name <你的配置名>
   ```
4. 启动训练：
   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <你的配置名> \
       --exp-name=my_experiment --overwrite
   ```

---

## 6. Isaac Lab 与 VLA 联调（仿真闭环评测）

本仓库 `bridge/isaac_lab_pi05_eval.py` 是核心桥接脚本：在 `isaac_sim_env` 中启动 Isaac Lab 任务，采集图像/状态，通过 WebSocket 发给 openpi 策略服务，拿回动作后驱动仿真机器人，形成"感知-决策-执行"闭环，并保存评测视频。

### 6.1 运行流程

**第一步：启动策略服务**（另开一个终端/GPU）

```bash
source ~/sim_stack/openpi_env/bin/activate
cd ~/sim_stack/openpi
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --env LIBERO --port 31437
```

**第二步：运行仿真评测桥接脚本**

```bash
source ~/sim_stack/isaac_sim_env/bin/activate
cd ~/sim_stack/IsaacLab

python ~/sim_stack/bridge/isaac_lab_pi05_eval.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
    --port 31437 \
    --prompt "stack the cubes" \
    --num_episodes 5 \
    --max_steps 200 \
    --headless \
    --video_out_path ~/sim_stack/videos
```

### 6.2 桥接脚本原理（供理解/二次开发）

1. 用 Isaac Lab 的 gym 接口创建任务环境，采集 `table_cam` / `wrist_cam` 图像与 `eef_pos`（末端位姿）等状态。
2. 调用 `openpi_client.websocket_client_policy.WebsocketClientPolicy.infer()` 将观测+语言指令发给远端策略服务。
3. 拿到一段动作 chunk（放入 deque 缓存），逐步 `env.step()` 执行，每 `--replan_steps` 步重新请求一次新的动作序列（滚动式重规划，兼顾响应速度与推理开销）。
4. 用 `imageio` 把每个 episode 的多路相机画面拼接保存为 mp4，便于人工检查效果。

---

## 7. 常见问题排查

| 现象                                                          | 可能原因                                       | 解决方法                                                                                                             |
| ------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `import isaacsim` 报错/卡住                                 | 未接受 EULA                                    | 加`OMNI_KIT_ACCEPT_EULA=YES` 环境变量                                                                              |
| Franka 资产加载路径报错                                       | beta 分支资产路径变更                          | 见 2.3 节的`sed` 修复命令                                                                                          |
| `CUDA illegal memory access`（双进程共享 GPU 时）           | JAX 默认预占大部分显存，与 Isaac Sim 冲突      | 用`CUDA_VISIBLE_DEVICES` 把两者分到不同 GPU；若必须共卡，设置 `XLA_PYTHON_CLIENT_MEM_FRACTION` 限制 JAX 显存占用 |
| `numpy` 版本冲突（openpi-client 装进 isaac_sim_env 后报错） | openpi-client 要 numpy<2.0，Isaac Sim 要 ≥2.x | 用`uv pip install --no-deps -e ...` 安装 openpi-client，不拉取其依赖                                               |
| 训练很慢                                                      | 未加`--headless`，或 `--num_envs` 太小     | 训练时始终加`--headless`；根据显存尽量调大并行环境数                                                               |
| 评测录像失败                                                  | 系统未装`ffmpeg`                             | `sudo apt install ffmpeg`                                                                                          |
| GLIBC 版本报错                                                | pip 安装 Isaac Sim 要求 GLIBC ≥ 2.35          | 升级系统或改用 Docker/Omniverse Launcher 二进制安装方式                                                              |

---

## 8. 参考资料

- Isaac Lab 官方文档：https://isaac-sim.github.io/IsaacLab/
- Isaac Lab GitHub：https://github.com/isaac-sim/IsaacLab
- Isaac Sim 官方文档：https://docs.omniverse.nvidia.com/isaacsim/latest/index.html
- openpi（Physical Intelligence）GitHub：https://github.com/Physical-Intelligence/openpi
- 本仓库内已有文档：`QUICKSTART.md`（速查）、`ISAAC_SIM_PI05_SETUP.md`（详细部署记录）
