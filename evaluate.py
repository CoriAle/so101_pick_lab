"""Evaluate a trained policy: success rate over N episodes, optionally record a video.
(PROVIDED; this is also how your result is graded.)

    python evaluate.py runs/first_try/best_model.zip --episodes 50 --randomization 0
    python evaluate.py runs/first_try/best_model.zip --episodes 5 --video rollout.mp4
    python evaluate.py runs/first_try/best_model.zip --json fixed.json

If `vecnormalize_best.pkl` / `vecnormalize_final.pkl` sits next to best_model.zip /
final_model.zip it is loaded automatically (frozen statistics, raw rewards).
Actions are always deterministic. Fixed-position (--randomization 0) and random-position
results must be reported separately.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import so101_pick  # noqa: F401
from so101_pick.env import LIFT_HEIGHT


class TerminalFrameCapture(gym.Wrapper):
    """Render the last frame *before* the vectorised env auto-resets."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["terminal_frame"] = self.env.render()
        return obs, reward, terminated, truncated, info


def load(path):
    for cls in (SAC, PPO):
        try:
            return cls.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path} as SAC or PPO")


def paired_normalization(model_path: str):
    """vecnormalize_<best|final>.pkl next to <best|final>_model.zip, if present."""
    p = Path(model_path)
    suffix = {"best_model": "best", "final_model": "final"}.get(p.stem)
    candidate = p.parent / f"vecnormalize_{suffix}.pkl" if suffix else None
    return str(candidate) if candidate and candidate.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--randomization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None, help="write the first few episodes to this .mp4")
    ap.add_argument("--vecnormalize", default=None)
    ap.add_argument("--json", dest="json_path", default=None, help="save per-episode measurements")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    if args.episodes <= 0:
        ap.error("--episodes must be positive")
    torch.set_num_threads(args.torch_threads)
    args.vecnormalize = args.vecnormalize or paired_normalization(args.model)

    render = args.video is not None

    def make_env():
        env = gym.make("SO101Pick-v0", randomization=args.randomization,
                       render_mode="rgb_array" if render else None)
        return TerminalFrameCapture(env) if render else env

    venv = DummyVecEnv([make_env])
    if args.vecnormalize:
        venv = VecNormalize.load(args.vecnormalize, venv)
        venv.training, venv.norm_reward = False, False   # frozen stats, raw rewards
        print("normalisation:", args.vecnormalize)
    else:
        print("normalisation: none")
    model = load(args.model)
    venv.seed(args.seed)

    successes, grasped, frames, rows = 0, 0, [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        done, ever_grasped = False, False
        peak, peak_grasp, min_dist, steps, floor_steps, ret = 0.0, 0.0, float("inf"), 0, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done_arr, infos = venv.step(action)
            done, info = bool(done_arr[0]), infos[0]
            steps += 1
            ret += float(rewards[0])
            ever_grasped |= bool(info.get("is_grasped", False))
            floor_steps += bool(info.get("arm_floor_contact", False))
            peak = max(peak, float(info["cube_height"]))
            if info["is_grasped"]:
                peak_grasp = max(peak_grasp, float(info["cube_height"]))
            min_dist = min(min_dist, float(info["dist_ee_cube"]))
            if render and ep < 5:
                frames.append(info["terminal_frame"] if done else venv.envs[0].unwrapped.render())
        ok = bool(info.get("is_success", False))
        successes += ok
        grasped += ever_grasped
        rows.append(dict(success=ok, grasped=ever_grasped, peak_height=peak, peak_grasp_height=peak_grasp,
                         min_distance=min_dist, steps=steps, floor_steps=floor_steps, episode_return=ret))

    n = args.episodes
    print(f"episodes: {n}  (randomization {args.randomization})")
    print(f"success rate (grasped + lifted {LIFT_HEIGHT * 100:g} cm): {successes / n:.1%}")
    print(f"grasped at least once:                {grasped / n:.1%}")
    print(f"mean peak height while grasped:       {100 * np.mean([r['peak_grasp_height'] for r in rows]):.2f} cm")
    print(f"mean steps: {np.mean([r['steps'] for r in rows]):.1f}   "
          f"episodes with arm-floor contact: {sum(r['floor_steps'] > 0 for r in rows)}")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(dict(
            model=args.model, vecnormalize=args.vecnormalize, target_height=LIFT_HEIGHT,
            randomization=args.randomization, seed=args.seed, success_rate=successes / n,
            grasp_rate=grasped / n, episodes=rows), indent=2) + "\n")
    if render and frames:
        imageio.mimsave(args.video, frames, fps=20)
        print("video written to", args.video)
    venv.close()


if __name__ == "__main__":
    main()
