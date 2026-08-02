# Isaac Sim + pi0.5 快速入门指南

面向：想快速把 Isaac Sim/Isaac Lab 仿真接上 pi0.5 (openpi) VLA 模型跑起来的人。只包含操作步骤和常用命令；背后的踩坑原因、原理细节见完整技术文档 [`ISAAC_SIM_PI05_SETUP.md`](./ISAAC_SIM_PI05_SETUP.md)。

## 0. 前置条件

- Ubuntu 22.04，NVIDIA GPU（建议 2 张，1 张跑仿真渲染、1 张跑模型推理；单卡也可以但要注意显存，见 §5）
- 已安装 `uv`（没有的话：`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- 能访问外网（下载 Isaac Sim 包、pi0.5 checkpoint，checkpoint 约 11.6GB）

## 1. 一次性安装

```bash
mkdir -p ~/sim_stack && cd ~/sim_stack

# 1.1 Isaac Sim（Python 3.12 环境）
uv venv isaac_sim_env --python 3.12
source isaac_sim_env/bin/activate
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease allow
OMNI_KIT_ACCEPT_EULA=YES python -c "import isaacsim"   # 首次运行同意 EULA
deactivate

# 1.2 Isaac Lab（装进同一个 3.12 环境，注意分支要对应 Isaac Sim 6.0.1）
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout release/3.0.0-beta2
source ~/sim_stack/isaac_sim_env/bin/activate
./isaaclab.sh --install
deactivate
cd ..

# 1.3 openpi / pi0.5（独立的 Python 3.11 环境）
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
UV_PROJECT_ENVIRONMENT=~/sim_stack/openpi_env uv sync
cd ..

# 1.4 openpi-client 装进 Isaac Sim 环境（用于桥接脚本调用 websocket）
source ~/sim_stack/isaac_sim_env/bin/activate
uv pip install --no-deps -e ~/sim_stack/openpi/packages/openpi-client
deactivate
```

**已知资产路径 bug**（Isaac Lab beta 分支的云资产库路径尚未同步），需要手动打一处补丁：

```bash
sed -i 's#Robots/FrankaEmika/panda_instanceable.usd#Robots/FrankaEmika/Legacy/panda_instanceable.usd#' \
    ~/sim_stack/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py
```

## 2. 获取桥接脚本

把 `~/sim_stack/bridge/isaac_lab_pi05_eval.py` 放到位（已有的话跳过）。这个脚本负责：Isaac Lab 出图像+状态 → websocket 发给 pi0.5 server → 拿动作写回仿真 → 存双相机视频。

## 3. 启动 pi0.5 policy server

```bash
source ~/sim_stack/openpi_env/bin/activate
cd ~/sim_stack/openpi
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python scripts/serve_policy.py --env LIBERO --port 31437
```

- 首次运行自动下载 `pi05_libero` checkpoint（约 11.6GB），之后走本地缓存 `~/.cache/openpi/`。
- 看到 `server listening on 0.0.0.0:31437` 即启动成功，保持这个终端常驻运行。
- `CUDA_VISIBLE_DEVICES=1` 把模型服务限定在一张卡上，把另一张卡留给仿真渲染，避免显存抢占导致仿真崩溃（详见 §5）。
- 端口号可自定义；如果是共享机器，先 `ss -tlnp | grep <port>` 确认没被占用。

## 4. 跑仿真 + 推理，存视频

另开一个终端：

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

跑完会打印每个 episode 的步数/是否成功，视频存到 `~/sim_stack/videos/episode_<N>_<success|failure>_table_wrist.mp4`（双相机左右拼接）。

**常用参数：**

| 参数 | 作用 |
|---|---|
| `--task` | Isaac Lab 任务 ID，换任务改这个 |
| `--num_episodes` | 跑几条 episode |
| `--max_steps` | 每条 episode 最长步数 |
| `--replan_steps` | 每几步重新问一次模型（默认 5，越小越"实时"但推理更频繁）|
| `--prompt` | 给模型的自然语言指令 |
| `--video_out_path` | 视频保存目录 |

## 5. 常见坑速查

| 现象 | 原因 | 解法 |
|---|---|---|
| headless 下卡住不动，无输出 | 误设了 `--visualizer kit`（需要显示设备） | headless 场景不要传 `--visualizer`，或显式传 `--visualizer none` |
| 相机读数时报 CUDA illegal memory access / 进程崩溃 | 仿真和模型 server 抢显存 | 用 `CUDA_VISIBLE_DEVICES` 把两者分到不同 GPU，JAX 进程加 `XLA_PYTHON_CLIENT_MEM_FRACTION` 限制预分配比例 |
| `import isaacsim` 卡住 | 首次运行要交互确认 EULA | 加环境变量 `OMNI_KIT_ACCEPT_EULA=YES` |
| Isaac Lab 报 `No module named 'omni.physics.tensors.impl'` | Isaac Lab 分支版本和 Isaac Sim 版本不匹配 | 确认用的是 `release/3.0.0-beta2`（对应 Isaac Sim 6.0.1），不要用 main 分支（对应 5.1.0）|
| 起 Franka 环境报 USD 资产 404 | Isaac Lab beta 分支资产路径滞后 | 见 §1 的 `sed` 补丁 |
| `isaacsim[all]` 装不上，报依赖冲突 | uv 跨 index 解析问题 | 显式锁版本号 + `--index-strategy unsafe-best-match --prerelease allow` |

## 6. 想换任务/模型怎么办

- **换 Isaac Lab 任务**：先列出所有任务名（见完整文档 §7.3），挑一个带 `Visuomotor` 的（有相机），改 `--task` 参数即可，脚本不用改。
- **换 pi0.5 checkpoint**（比如自己微调后的）：把 `serve_policy.py --env LIBERO` 换成 `serve_policy.py policy:checkpoint --policy.config=<config名> --policy.dir=<ckpt路径>`，桥接脚本这边不用动，只要 `--port` 对得上。
- **对 pi0.5 做微调**：简要流程是"采数据转 LeRobot 格式 → 写数据映射配置 → `compute_norm_stats.py` → `train.py` → 微调后 checkpoint 起 server"，完整步骤和硬件要求见完整技术文档 §9。

## 7. 用 Isaac Lab 做强化学习训练（比如训练机械狗运动）

Isaac Lab 自带 Unitree Go1/Go2/A1、ANYmal、Cassie、H1/G1 等足式机器人的 velocity locomotion RL 任务，不用写代码，直接：

```bash
source ~/sim_stack/isaac_sim_env/bin/activate
cd ~/sim_stack/IsaacLab

# 训练（以 Go2 平地速度跟踪为例）
./isaaclab.sh train --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless --num_envs 4096 --max_iterations 300

# checkpoint 存到 logs/rsl_rl/unitree_go2_flat/<时间戳>/model_<iter>.pt

# 评测/录视频
./isaaclab.sh play --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --headless --num_envs 4 \
    --checkpoint logs/rsl_rl/unitree_go2_flat/<时间戳>/model_<iter>.pt \
    --video --video_length 200
```

换机器人/地形只需要换 `--task`（如 `Isaac-Velocity-Rough-Unitree-Go2-v0` 是起伏地形，`Anymal-C`/`Cassie`/`H1` 等同理）；调超参数改对应任务目录下的 `agents/rsl_rl_ppo_cfg.py`。任务设计（观测/动作/奖励/终止条件）、支持的 RL 库列表、从零新增机器人任务的路径，完整说明见技术文档 §10。

## 8. 更多细节

- 完整原理、每个坑的排查过程、pi0.5 推理/微调机制、RL 训练机制详解：见 [`ISAAC_SIM_PI05_SETUP.md`](./ISAAC_SIM_PI05_SETUP.md)。
