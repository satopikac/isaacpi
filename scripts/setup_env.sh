#!/usr/bin/env bash
#
# Isaac Sim 6.0.1 + Isaac Lab release/3.0.0-beta2 + openpi (pi0.5) 环境自动配置脚本
#
# 用法：
#   ./scripts/setup_env.sh                 # 完整安装
#   ./scripts/setup_env.sh --verify        # 只做体检，不装任何东西
#   ./scripts/setup_env.sh --only isaac    # 只装 Isaac Sim + Isaac Lab
#   ./scripts/setup_env.sh --only openpi   # 只装 openpi
#   ./scripts/setup_env.sh --root /data/sim_stack
#
# 脚本是幂等的：每一步都会先检测是否已完成，重复执行只补做缺失的部分。
# 全部步骤来自 ISAAC_SIM_PI05_SETUP.md §2 与 QUICKSTART.md §1 的真实安装记录，
# 文档中记录的每个坑（版本锁定、EULA、分支匹配、numpy 冲突、资产路径）都已固化在下面。

set -euo pipefail

# ---------------------------------------------------------------- 配置项

SIM_STACK_ROOT="${SIM_STACK_ROOT:-$HOME/sim_stack}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ISAACSIM_VERSION="6.0.1.0"
ISAACLAB_BRANCH="release/3.0.0-beta2"   # 必须匹配 Isaac Sim 6.0.1，main 分支对应 5.1.0
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi"
PY_ISAAC="3.12"                          # Isaac Sim 6.0.1 要求
PY_OPENPI="3.11"                         # openpi 的 .python-version
REQUIRED_DISK_GB=45                      # 两个 env 约 39G + clone + 余量

MODE="install"
ONLY=""
ASSUME_YES=0

# ---------------------------------------------------------------- 输出工具

if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
    C_BLU=$'\033[34m'; C_DIM=$'\033[2m';  C_RST=$'\033[0m'; C_BLD=$'\033[1m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_DIM=""; C_RST=""; C_BLD=""
fi

STEP_NO=0
step()  { STEP_NO=$((STEP_NO + 1)); printf '\n%s[%d] %s%s\n' "$C_BLU$C_BLD" "$STEP_NO" "$*" "$C_RST"; }
ok()    { printf '  %s✓%s %s\n' "$C_GRN" "$C_RST" "$*"; }
skip()  { printf '  %s·%s %s %s(已存在，跳过)%s\n' "$C_DIM" "$C_RST" "$*" "$C_DIM" "$C_RST"; }
warn()  { printf '  %s!%s %s\n' "$C_YEL" "$C_RST" "$*" >&2; }
die()   { printf '\n%s✗ 失败：%s%s\n' "$C_RED$C_BLD" "$*" "$C_RST" >&2; exit 1; }
info()  { printf '  %s%s%s\n' "$C_DIM" "$*" "$C_RST"; }

confirm() {
    [[ $ASSUME_YES -eq 1 ]] && return 0
    local reply
    read -r -p "  ${C_YEL}?${C_RST} $1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------- 参数解析

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)  MODE="verify"; shift ;;
        --only)    ONLY="${2:-}"; shift 2 ;;
        --root)    SIM_STACK_ROOT="${2:-}"; shift 2 ;;
        -y|--yes)  ASSUME_YES=1; shift ;;
        -h|--help) awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; exit 0 ;;
        *)         die "未知参数：$1（用 --help 看用法）" ;;
    esac
done

[[ -n "$ONLY" && "$ONLY" != "isaac" && "$ONLY" != "openpi" ]] && \
    die "--only 只接受 isaac 或 openpi，收到：$ONLY"

want_isaac()  { [[ -z "$ONLY" || "$ONLY" == "isaac"  ]]; }
want_openpi() { [[ -z "$ONLY" || "$ONLY" == "openpi" ]]; }

ISAAC_ENV="$SIM_STACK_ROOT/isaac_sim_env"
OPENPI_ENV="$SIM_STACK_ROOT/openpi_env"
ISAACLAB_DIR="$SIM_STACK_ROOT/IsaacLab"
OPENPI_DIR="$SIM_STACK_ROOT/openpi"
FRANKA_CFG="$ISAACLAB_DIR/source/isaaclab_assets/isaaclab_assets/robots/franka.py"

# ---------------------------------------------------------------- 体检模式

check_one() {
    # $1=描述  $2=命令
    if eval "$2" >/dev/null 2>&1; then
        printf '  %s✓%s %s\n' "$C_GRN" "$C_RST" "$1"; return 0
    else
        printf '  %s✗%s %s\n' "$C_RED" "$C_RST" "$1"; return 1
    fi
}

# 分支必须是 release/3.0.0-beta2（对应 Isaac Sim 6.0.1）。
# main 分支对应 5.1.0，会在 import isaaclab_tasks 时报 omni.physics.tensors.impl 缺失。
# 允许 detached HEAD，但要求该提交确实落在目标分支上。
isaaclab_branch_ok() {
    local dir="$1" want="$2" cur
    [[ -d "$dir/.git" ]] || return 1
    cur="$(git -C "$dir" symbolic-ref -q --short HEAD 2>/dev/null || echo '')"
    [[ "$cur" == "$want" ]] && return 0
    git -C "$dir" branch -a --contains HEAD 2>/dev/null \
        | sed 's#^[* ]*##; s#^remotes/[^/]*/##' \
        | grep -qx "$want"
}

run_verify() {
    local fail=0
    printf '\n%s环境体检%s  (root: %s)\n' "$C_BLD" "$C_RST" "$SIM_STACK_ROOT"

    printf '\n%s前置工具%s\n' "$C_BLD" "$C_RST"
    check_one "uv 已安装"            "command -v uv"                || fail=1
    check_one "git 已安装"           "command -v git"               || fail=1
    check_one "nvidia-smi 可用"      "command -v nvidia-smi"        || fail=1

    printf '\n%sIsaac Sim / Isaac Lab%s\n' "$C_BLD" "$C_RST"
    check_one "isaac_sim_env 存在"   "test -x '$ISAAC_ENV/bin/python'"  || fail=1
    check_one "isaacsim 可导入"      "OMNI_KIT_ACCEPT_EULA=YES '$ISAAC_ENV/bin/python' -c 'import isaacsim'" || fail=1
    check_one "IsaacLab 已 clone"    "test -d '$ISAACLAB_DIR/.git'"     || fail=1
    check_one "IsaacLab 分支正确"    "isaaclab_branch_ok '$ISAACLAB_DIR' '$ISAACLAB_BRANCH'" || fail=1
    check_one "isaaclab 可导入"      "'$ISAAC_ENV/bin/python' -c 'import isaaclab'" || fail=1
    check_one "franka 资产路径已修复" "grep -q 'FrankaEmika/Legacy/panda_instanceable.usd' '$FRANKA_CFG'" || fail=1
    check_one "openpi_client 已装入"  "'$ISAAC_ENV/bin/python' -c 'import openpi_client'" || fail=1

    printf '\n%sopenpi / pi0.5%s\n' "$C_BLD" "$C_RST"
    check_one "openpi_env 存在"      "test -x '$OPENPI_ENV/bin/python'" || fail=1
    check_one "openpi 已 clone"      "test -d '$OPENPI_DIR/.git'"       || fail=1
    check_one "jax 可导入"           "'$OPENPI_ENV/bin/python' -c 'import jax'" || fail=1
    check_one "jax 能看到 GPU"       "'$OPENPI_ENV/bin/python' -c 'import jax,sys; sys.exit(0 if any(d.platform==\"gpu\" for d in jax.devices()) else 1)'" || fail=1

    printf '\n%s桥接%s\n' "$C_BLD" "$C_RST"
    check_one "桥接脚本已就位"        "test -f '$SIM_STACK_ROOT/bridge/isaac_lab_pi05_eval.py'" || fail=1

    if [[ $fail -eq 0 ]]; then
        printf '\n%s全部通过。%s下一步见 QUICKSTART.md §3。\n\n' "$C_GRN$C_BLD" "$C_RST"
    else
        printf '\n%s有未通过项。%s跑 %s./scripts/setup_env.sh%s 补齐。\n\n' \
            "$C_YEL$C_BLD" "$C_RST" "$C_BLD" "$C_RST"
    fi
    return $fail
}

if [[ "$MODE" == "verify" ]]; then
    run_verify
    exit $?
fi

# ---------------------------------------------------------------- 0. 前置检查

printf '\n%s╭─ Isaac Sim + Isaac Lab + pi0.5 环境配置 ─╮%s\n' "$C_BLD" "$C_RST"
info "安装位置：$SIM_STACK_ROOT"
info "本仓库：  $REPO_DIR"

step "前置检查"

[[ "$(uname -s)" == "Linux" ]] || die "本脚本只支持 Linux（文档基于 Ubuntu 22.04 验证）"

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" != "22.04" ]]; then
        warn "文档在 Ubuntu 22.04 上验证，当前是 ${PRETTY_NAME:-未知}，可能有差异"
    else
        ok "系统：${PRETTY_NAME:-Linux}"
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    ok "检测到 $GPU_COUNT 张 GPU："
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/      /'
    if [[ "$GPU_COUNT" -lt 2 ]]; then
        warn "只有 1 张 GPU。仿真渲染与 pi0.5 推理会争抢显存，可能触发 CUDA illegal memory access。"
        warn "单卡请参考 QUICKSTART.md §5，用 XLA_PYTHON_CLIENT_MEM_FRACTION 限制 JAX 预分配。"
    fi
else
    warn "没找到 nvidia-smi。Isaac Sim 需要 NVIDIA GPU 与驱动才能跑起来。"
    confirm "仍然继续？" || exit 1
fi

# 磁盘空间：两个虚拟环境实测约 39G（isaac_sim_env ~31G + openpi_env ~7.7G）
mkdir -p "$SIM_STACK_ROOT"
AVAIL_GB="$(df -BG --output=avail "$SIM_STACK_ROOT" | tail -1 | tr -dc '0-9')"
if [[ "$AVAIL_GB" -lt "$REQUIRED_DISK_GB" ]]; then
    warn "可用空间 ${AVAIL_GB}G，建议至少 ${REQUIRED_DISK_GB}G（两个 env 实测约 39G，不含 11.6G checkpoint）"
    confirm "仍然继续？" || exit 1
else
    ok "磁盘可用 ${AVAIL_GB}G（需要约 ${REQUIRED_DISK_GB}G）"
fi

command -v git >/dev/null 2>&1 || die "需要 git，请先安装：sudo apt install git"

if ! command -v uv >/dev/null 2>&1; then
    warn "未检测到 uv（用于管理 Python 版本与虚拟环境，无需 sudo）"
    if confirm "现在安装 uv？"; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null 2>&1 || die "uv 安装后仍不可用，请重开终端后重试"
        ok "uv 安装完成：$(uv --version)"
    else
        die "uv 是必需的，见 https://astral.sh/uv"
    fi
else
    ok "uv 已就绪：$(uv --version)"
fi

# ---------------------------------------------------------------- 1. Isaac Sim

if want_isaac; then

step "创建 Isaac Sim 虚拟环境（Python $PY_ISAAC）"
if [[ -x "$ISAAC_ENV/bin/python" ]]; then
    skip "$ISAAC_ENV"
else
    uv venv "$ISAAC_ENV" --python "$PY_ISAAC"
    ok "已创建 $ISAAC_ENV"
fi

step "安装 Isaac Sim $ISAACSIM_VERSION"
if "$ISAAC_ENV/bin/python" -c 'import isaacsim' >/dev/null 2>&1; then
    skip "isaacsim 已安装"
else
    info "从 pypi.nvidia.com 拉取，包很大，通常需要 10-30 分钟…"
    # 三个 flag 都是必需的，缺一个就装不上（原因见 ISAAC_SIM_PI05_SETUP.md §2.2 踩坑表）：
    #   ==6.0.1.0                     锁死版本，否则 resolver 会退到 5.0.0.0 并构建失败
    #   --index-strategy unsafe-best-match  允许跨 index 解析，解决 mujoco-usd-converter 冲突
    #   --prerelease allow            tinyobjloader==2.0.0rc13 是预发布版
    uv pip install --python "$ISAAC_ENV/bin/python" \
        "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match \
        --prerelease allow
    ok "isaacsim $ISAACSIM_VERSION 安装完成"
fi

step "接受 Omniverse EULA 并验证导入"
# 首次 import 需要交互式同意 EULA，OMNI_KIT_ACCEPT_EULA=YES 可跳过卡住
if OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_ENV/bin/python" -c 'import isaacsim' >/dev/null 2>&1; then
    ok "isaacsim 可正常导入"
else
    die "isaacsim 导入失败，请手动排查：OMNI_KIT_ACCEPT_EULA=YES $ISAAC_ENV/bin/python -c 'import isaacsim'"
fi

# ---------------------------------------------------------------- 2. Isaac Lab

step "获取 Isaac Lab（$ISAACLAB_BRANCH）"
if [[ -d "$ISAACLAB_DIR/.git" ]]; then
    skip "$ISAACLAB_DIR"
else
    git clone "$ISAACLAB_REPO" "$ISAACLAB_DIR"
    ok "已 clone 到 $ISAACLAB_DIR"
fi

# 分支必须对应 Isaac Sim 6.0.1；main 分支对应 5.1.0，会报
# ModuleNotFoundError: No module named 'omni.physics.tensors.impl'
if isaaclab_branch_ok "$ISAACLAB_DIR" "$ISAACLAB_BRANCH"; then
    ok "已在 $ISAACLAB_BRANCH"
else
    CURRENT_REF="$(git -C "$ISAACLAB_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    info "当前在 '$CURRENT_REF'，切换到 $ISAACLAB_BRANCH（需匹配 Isaac Sim 6.0.1）"
    git -C "$ISAACLAB_DIR" fetch origin "$ISAACLAB_BRANCH" --depth 1 2>/dev/null || git -C "$ISAACLAB_DIR" fetch origin
    git -C "$ISAACLAB_DIR" checkout "$ISAACLAB_BRANCH"
    ok "已切换到 $ISAACLAB_BRANCH"
fi

step "安装 Isaac Lab 到 isaac_sim_env"
if "$ISAAC_ENV/bin/python" -c 'import isaaclab' >/dev/null 2>&1; then
    skip "isaaclab 已安装"
else
    info "运行 ./isaaclab.sh --install，需要几分钟…"
    # isaaclab.sh 依赖激活态的 venv 来定位 python
    (
        # shellcheck disable=SC1091
        source "$ISAAC_ENV/bin/activate"
        cd "$ISAACLAB_DIR"
        ./isaaclab.sh --install
    )
    ok "Isaac Lab 安装完成"
fi

step "修复 Franka USD 资产路径"
# beta 分支资产迁移滞后：6.0 资产库里该文件已移到 Legacy/ 子目录，不修会 404
if [[ ! -f "$FRANKA_CFG" ]]; then
    warn "找不到 $FRANKA_CFG，跳过（上游可能已调整目录结构）"
elif grep -q 'FrankaEmika/Legacy/panda_instanceable.usd' "$FRANKA_CFG"; then
    skip "资产路径已是 Legacy/"
else
    if git -C "$ISAACLAB_DIR" apply --check "$REPO_DIR/patches/isaaclab-franka-legacy-usd-path.patch" 2>/dev/null; then
        git -C "$ISAACLAB_DIR" apply "$REPO_DIR/patches/isaaclab-franka-legacy-usd-path.patch"
        ok "已应用 patches/isaaclab-franka-legacy-usd-path.patch"
    else
        # patch 上下文对不上（上游改过该文件）时退回 sed
        sed -i 's#Robots/FrankaEmika/panda_instanceable.usd#Robots/FrankaEmika/Legacy/panda_instanceable.usd#' "$FRANKA_CFG"
        grep -q 'FrankaEmika/Legacy/panda_instanceable.usd' "$FRANKA_CFG" \
            && ok "已用 sed 修复资产路径" \
            || warn "自动修复失败，请手动检查 $FRANKA_CFG"
    fi
fi

fi  # want_isaac

# ---------------------------------------------------------------- 3. openpi

if want_openpi; then

step "获取 openpi"
if [[ -d "$OPENPI_DIR/.git" ]]; then
    skip "$OPENPI_DIR"
else
    git clone "$OPENPI_REPO" "$OPENPI_DIR"
    ok "已 clone 到 $OPENPI_DIR"
fi

step "创建 openpi 虚拟环境（Python $PY_OPENPI）并同步依赖"
if [[ -x "$OPENPI_ENV/bin/python" ]] && "$OPENPI_ENV/bin/python" -c 'import jax' >/dev/null 2>&1; then
    skip "openpi_env 已就绪"
else
    [[ -x "$OPENPI_ENV/bin/python" ]] || uv venv "$OPENPI_ENV" --python "$PY_OPENPI"
    info "uv sync 按 uv.lock 精确安装 jax[cuda12]==0.5.3 / torch==2.7.1 等，需要几分钟…"
    (
        cd "$OPENPI_DIR"
        UV_PROJECT_ENVIRONMENT="$OPENPI_ENV" uv sync
    )
    ok "openpi 依赖安装完成"
fi

fi  # want_openpi

# ---------------------------------------------------------------- 4. 桥接

if [[ -z "$ONLY" ]]; then

step "把 openpi-client 装进 isaac_sim_env"
# openpi_client 的 pyproject 要求 numpy<2.0.0，而 Isaac Sim 生态要 numpy>=2.x，
# 必须 --no-deps 单独装，否则会把 Isaac Sim 的 numpy 降级搞坏
if "$ISAAC_ENV/bin/python" -c 'import openpi_client' >/dev/null 2>&1; then
    skip "openpi_client 已可导入"
elif [[ -d "$OPENPI_DIR/packages/openpi-client" ]]; then
    uv pip install --python "$ISAAC_ENV/bin/python" --no-deps -e "$OPENPI_DIR/packages/openpi-client"
    ok "openpi_client 已装入（--no-deps，保护 numpy 版本）"
else
    warn "找不到 $OPENPI_DIR/packages/openpi-client，跳过"
fi

step "部署桥接脚本"
mkdir -p "$SIM_STACK_ROOT/bridge" "$SIM_STACK_ROOT/videos"
if [[ "$REPO_DIR/bridge/isaac_lab_pi05_eval.py" -ef "$SIM_STACK_ROOT/bridge/isaac_lab_pi05_eval.py" ]]; then
    skip "桥接脚本就在原位"
else
    cp "$REPO_DIR/bridge/isaac_lab_pi05_eval.py" "$SIM_STACK_ROOT/bridge/"
    ok "已复制到 $SIM_STACK_ROOT/bridge/"
fi

fi  # 桥接

# ---------------------------------------------------------------- 完成

printf '\n%s╰─ 安装完成 ─╯%s\n' "$C_GRN$C_BLD" "$C_RST"
printf '\n体检：%s./scripts/setup_env.sh --verify%s\n' "$C_BLD" "$C_RST"

cat <<EOF

${C_BLD}下一步${C_RST}（详见 QUICKSTART.md §3–4）

  ${C_DIM}# 终端 1：启动 pi0.5 policy server（首次会下载 11.6G checkpoint）${C_RST}
  source $OPENPI_ENV/bin/activate
  cd $OPENPI_DIR
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
      python scripts/serve_policy.py --env LIBERO --port 31437

  ${C_DIM}# 终端 2：跑仿真闭环${C_RST}
  source $ISAAC_ENV/bin/activate
  cd $ISAACLAB_DIR
  python $SIM_STACK_ROOT/bridge/isaac_lab_pi05_eval.py \\
      --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \\
      --port 31437 --prompt "stack the cubes" \\
      --num_episodes 5 --max_steps 200 --headless \\
      --video_out_path $SIM_STACK_ROOT/videos

  ${C_DIM}# 或者：跑 Go2 强化学习训练（见 ISAAC_LAB_RL_TRAINING_PRACTICE.md）${C_RST}
  source $ISAAC_ENV/bin/activate && cd $ISAACLAB_DIR
  ./isaaclab.sh train --rl_library rsl_rl \\
      --task Isaac-Velocity-Flat-Unitree-Go2-v0 \\
      --headless --num_envs 4096 --max_iterations 300

EOF
