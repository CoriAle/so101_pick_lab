"""Final model selection on a held-out VALIDATION set (run once, after training).

Compares the checkpoints of a run (best_model = chosen during training on 20 eval episodes,
final_model = last weights) on `--episodes` random cube positions drawn with `--val-seed`.
That seed differs from the in-training eval seed (1000) and from the TEST seeds used for
reporting (1 and 2), so the reported numbers are not used to pick the model.

The winner (higher success rate, ties broken by mean episode return) is written as
best_model.zip + vecnormalize_best.pkl. The in-training checkpoint is kept as
checkpoint_eval20_model.zip + vecnormalize_checkpoint_eval20.pkl. A selection.json records everything.

    python scripts/select_model.py runs/<run-name>
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import so101_pick  # noqa: E402,F401


def validate(model_path: Path, norm_path: Path, episodes: int, seed: int, randomization: float):
    venv = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: gym.make("SO101Pick-v0",
                                                                           randomization=randomization)]))
    venv.training, venv.norm_reward = False, False
    model = SAC.load(str(model_path), device="cpu")
    venv.seed(seed)
    successes, returns = [], []
    for _ in range(episodes):
        obs, done, ret = venv.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, infos = venv.step(action)
            done, ret = bool(d[0]), ret + float(r[0])
        successes.append(bool(infos[0]["is_success"]))
        returns.append(ret)
    venv.close()
    return float(np.mean(successes)), float(np.mean(returns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--val-seed", type=int, default=500)
    ap.add_argument("--randomization", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run_dir)
    if (run / "selection.json").exists():
        sys.exit(f"{run} was already selected (selection.json exists); refusing to select twice")

    shutil.move(run / "best_model.zip", run / "checkpoint_eval20_model.zip")
    shutil.move(run / "vecnormalize_best.pkl", run / "vecnormalize_checkpoint_eval20.pkl")
    candidates = {
        "checkpoint_eval20": (run / "checkpoint_eval20_model.zip", run / "vecnormalize_checkpoint_eval20.pkl"),
        "final": (run / "final_model.zip", run / "vecnormalize_final.pkl"),
    }
    results = {}
    for name, (model_path, norm_path) in candidates.items():
        success, ret = validate(model_path, norm_path, args.episodes, args.val_seed, args.randomization)
        results[name] = {"success_rate": success, "mean_return": ret}
        print(f"{name:18s} validation success {success:.1%}  mean return {ret:.1f}")

    winner = max(results, key=lambda k: (results[k]["success_rate"], results[k]["mean_return"]))
    shutil.copy(candidates[winner][0], run / "best_model.zip")
    shutil.copy(candidates[winner][1], run / "vecnormalize_best.pkl")
    (run / "selection.json").write_text(json.dumps({
        "rule": "highest validation success rate, ties broken by mean episode return",
        "validation": {"episodes": args.episodes, "seed": args.val_seed, "randomization": args.randomization},
        "not_used_for_selection": "test seeds 1 and 2 (reported results)",
        "results": results, "selected": winner,
        "best_model.zip": f"copy of {candidates[winner][0].name}",
    }, indent=2) + "\n")
    print(f"selected: {winner} -> best_model.zip + vecnormalize_best.pkl")


if __name__ == "__main__":
    main()
