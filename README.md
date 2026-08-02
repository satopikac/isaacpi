# Isaac Sim / Isaac Lab + pi0.5 实战笔记

从零在一台裸机 Ubuntu 上把 **NVIDIA Isaac Sim / Isaac Lab** 仿真平台搭起来，跑通两条主线：

1. **机器人强化学习**——训练 Unitree Go2 四足机器人平地行走，出训练曲线和演示视频；
2. **VLA 模型闭环**——接入 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi) / **pi0.5** 模型，实现 `Isaac Lab 出图像+状态 → websocket → pi0.5 推理 → 动作写回仿真` 的端到端闭环。

所有文档都是**实际操作记录**，包含真实执行的命令、真实输出、遇到的每一个报错和对应的根因与解法，而不是照抄官方文档。

## 环境基线

本仓库全部内容在以下环境验证通过：

| 项 | 版本 |
| --- | --- |
| 系统 | Ubuntu 22.04，无 docker |
| GPU | 双 NVIDIA RTX 5080 |
| 驱动 / CUDA | 580.126.20 / CUDA 13.0 |
| Isaac Sim | 6.0.1.0（Python 3.12 环境） |
| Isaac Lab | `release/3.0.0-beta2` |
| openpi / pi0.5 | 独立 Python 3.11 环境，`pi05_libero` checkpoint（约 11.6 GiB） |

## 文档导读

建议按下面的顺序读。**没有基础就从 1 开始，只想赶紧跑起来就直接看 2。**

| # | 文档 | 内容 |
| --- | --- | --- |
| 1 | [ISAAC_完整入门指南.md](./ISAAC_完整入门指南.md) | **零基础入口**。核心概念与架构、安装、仿真基础、RL 训练、VLA 部署与联调、排错，一篇覆盖全景 |
| 2 | [QUICKSTART.md](./QUICKSTART.md) | **速查手册**。只有操作步骤和命令，不讲原理，照着敲就能跑 |
| 3 | [ISAAC_MANUAL.md](./ISAAC_MANUAL.md) | **完整说明书**（最厚的一本）。是 1 和 4 的合并升级版，每个步骤都补了"这是什么、为什么这么设计"，含桥接脚本逐行代码解读 |
| 4 | [ISAAC_SIM_PI05_SETUP.md](./ISAAC_SIM_PI05_SETUP.md) | **原始技术记录**。环境搭建到端到端闭环的第一手过程记录 |
| 5 | [ISAAC_LAB_RL_TRAINING_PRACTICE.md](./ISAAC_LAB_RL_TRAINING_PRACTICE.md) | **RL 训练实战**。以 Go2 机械狗走路为例：启动训练、看懂日志、TensorBoard、录制视频，含完整踩坑清单和可复用命令模板 |
| 6 | [PI05_DEPLOYMENT_PRACTICE.md](./PI05_DEPLOYMENT_PRACTICE.md) | **pi0.5 实战手册**。checkpoint 复现、policy server 部署、websocket 通信协议详解、与真机/仿真通信、微调 |
| 7 | [RTC_实时动作分块推理_论文报告.md](./RTC_实时动作分块推理_论文报告.md) | **论文精读**。Real-Time Chunking（NeurIPS 2025, [arXiv:2506.07339](https://arxiv.org/abs/2506.07339)）——VLA 推理延迟怎么解决，以及对本仓库部署的启示 |

> 文档 3/4 和 1/2 内容有意重叠：4 是第一次做的原始记录，3 是补齐原理后的重写版。想看"当时到底发生了什么"读 4，想系统学读 3。

## 仓库结构

```
.
├── bridge/isaac_lab_pi05_eval.py   # Isaac Lab ↔ pi0.5 桥接脚本（本仓库核心代码）
├── patches/                        # 需要打在 Isaac Lab 上游代码上的补丁
├── videos/                         # 端到端运行产出的演示视频
├── logs/                           # Go2 训练/评测的真实日志
└── *.md                            # 上表中的 7 篇文档
```

`videos/` 里两个文件分别是：

- `go2_walk_demo_final.mp4`——Go2 强化学习训练完成后的行走演示；
- `episode_0_failure_table_wrist.mp4`——pi0.5 闭环的双相机记录，**这是一次失败案例**（LIBERO 上训练的 checkpoint 直接零样本迁移到 Isaac Lab 的 Franka 场景，动作并不合理），原因分析见 `ISAAC_SIM_PI05_SETUP.md` §5。

## 关于依赖

本仓库**只包含自己写的文档、脚本和产物**，不含 Isaac Lab、openpi 这两个上游仓库的代码，也不含 Python 虚拟环境（两个 env 加起来近 40 GB）。请按 [QUICKSTART.md §1](./QUICKSTART.md) 自行 clone 和安装。

Isaac Lab beta 分支有一处云资产路径未同步的 bug，需要打补丁：

```bash
git -C /path/to/IsaacLab apply /path/to/this/repo/patches/isaaclab-franka-legacy-usd-path.patch
```

## 许可

文档与本仓库自有代码以 MIT 协议发布，见 [LICENSE](./LICENSE)。Isaac Sim、Isaac Lab、openpi 各自遵循其上游许可。
