# Runs in the isaac_sim_env (Python 3.12) venv with Isaac Lab installed.
# Connects a running openpi policy server (pi0.5) to an Isaac Lab manipulation
# task over the websocket client-server protocol openpi already uses for LIBERO.

import argparse
import collections
import contextlib
import pathlib
import sys

import gymnasium as gym
import imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Evaluate a pi0.5 policy server against an Isaac Lab task.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--prompt", type=str, default="stack the cubes")
parser.add_argument("--replan_steps", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--resize_size", type=int, default=224)
parser.add_argument("--video_out_path", type=str, default="data/pi05_isaac_lab/videos")
add_launcher_args(parser)
parser.set_defaults(num_envs=1)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _to_uint8_image(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def main():
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    client = _websocket_client_policy.WebsocketClientPolicy(args_cli.host, args_cli.port)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        video_dir = pathlib.Path(args_cli.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)

        total_episodes, total_successes = 0, 0
        for episode_idx in range(args_cli.num_episodes):
            obs, _ = env.reset()
            action_plan = collections.deque()
            done = False
            t = 0
            table_frames, wrist_frames = [], []

            while t < args_cli.max_steps and not done:
                with torch.inference_mode():
                    policy_obs = obs["policy"]
                    img_raw = _to_uint8_image(policy_obs["table_cam"][0])
                    wrist_img_raw = _to_uint8_image(policy_obs["wrist_cam"][0])
                    table_frames.append(img_raw)
                    wrist_frames.append(wrist_img_raw)

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_raw, args_cli.resize_size, args_cli.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img_raw, args_cli.resize_size, args_cli.resize_size)
                    )
                    state = policy_obs["eef_pos"][0].cpu().numpy() if "eef_pos" in policy_obs else np.zeros(8)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": args_cli.prompt,
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args_cli.replan_steps, (
                            f"replan every {args_cli.replan_steps} steps but policy only predicts"
                            f" {len(action_chunk)} steps"
                        )
                        action_plan.extend(action_chunk[: args_cli.replan_steps])

                    action = action_plan.popleft()
                    action_t = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
                    obs, reward, terminated, truncated, info = env.step(action_t)
                    done = bool(terminated[0] or truncated[0])
                    t += 1

            total_episodes += 1
            if done:
                total_successes += 1
            print(f"[episode {episode_idx}] steps={t} success={done}")

            suffix = "success" if done else "failure"
            combined_frames = [np.concatenate([tf, wf], axis=1) for tf, wf in zip(table_frames, wrist_frames)]
            imageio.mimwrite(
                video_dir / f"episode_{episode_idx}_{suffix}_table_wrist.mp4",
                combined_frames,
                fps=10,
            )

        print(f"Success rate: {total_successes}/{total_episodes}")
        env.close()


if __name__ == "__main__":
    main()
