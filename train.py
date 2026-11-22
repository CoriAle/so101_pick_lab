"""Training script (PROVIDED harness + TODO 2: tune the hyperparameters).

Usage examples
--------------
    python train.py --run-name first_try --timesteps 200000 --randomization 0
    python train.py --run-name cont --init runs/first_try/best_model.zip \
        --vecnormalize runs/first_try/vecnormalize_best.pkl --timesteps 100000
    tensorboard --logdir runs

Outputs in runs/<run-name>/: config.json, a snapshot of the source files used,
best_model.zip + vecnormalize_best.pkl (always saved together), final_model.zip +
vecnormalize_final.pkl, TensorBoard logs.
"""
import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC  # noqa: F401
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import so101_pick  # noqa: F401  (registers "SO101Pick-v0")
from so101_pick.callbacks import (ResetCurriculumCallback, SaveNormalizationCallback,
                                  StageEvalCallback, StopOnFullSuccessCallback)
from so101_pick.env import LIFT_HEIGHT

# TODO 2a: algorithm. SAC is off-policy and sample-efficient: with only 20 Hz x 150-step
# episodes and one cube, reusing every transition from a replay buffer matters more than
# the wall-clock speed of PPO's many parallel envs.
ALGO = "SAC"

# TODO 2b: hyperparameters. Starting point = the peer reference configuration
# (smaller net and batch than the original baseline). Change one thing per run.
HPARAMS_SAC = {
    "learning_rate": 3e-4,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

HPARAMS_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

HPARAMS = HPARAMS_SAC if ALGO == "SAC" else HPARAMS_PPO

# TODO 2c: normalise observations (joint angles, velocities and metres live on different
# scales); rewards stay unnormalised so the success bonus keeps its meaning.
USE_VECNORMALIZE = True

N_ENVS = 4


def make_env(randomization: float):
    return lambda: gym.make("SO101Pick-v0", randomization=randomization)


def build_model(venv, tb_dir: str, algorithm: str = ALGO, hparams: dict | None = None):
    if algorithm not in ("SAC", "PPO"):
        raise ValueError("algorithm must be 'SAC' or 'PPO'")
    cls = {"SAC": SAC, "PPO": PPO}[algorithm]
    return cls("MlpPolicy", venv, tensorboard_log=tb_dir, verbose=1,
               **(HPARAMS if hparams is None else hparams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--randomization", type=float, default=0.0,
                    help="0 = cube always at the same place, 1 = fully random")
    ap.add_argument("--init", default=None, help="checkpoint .zip to CONTINUE from (not a from-scratch run)")
    ap.add_argument("--vecnormalize", default=None, help="stats paired with --init")
    ap.add_argument("--replay-buffer", default=None, help="SAC replay buffer to resume with --init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algorithm", choices=("SAC", "PPO"), default=ALGO)
    ap.add_argument("--n-envs", type=int, default=N_ENVS)
    ap.add_argument("--subproc", action="store_true", help="subprocess vectorisation (default: sequential)")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--eval-freq", type=int, default=5_000, help="env steps between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--curriculum-steps", type=int, default=50_000,
                    help="fade easy near-cube resets to zero over this many steps; 0 disables")
    ap.add_argument("--net-width", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction,
                    default=USE_VECNORMALIZE)
    ap.add_argument("--stop-after-success-evals", type=int, default=0,
                    help="stop after N consecutive 100%% evaluations; 0 disables")
    args = ap.parse_args()

    ints = [args.n_envs, args.torch_threads, args.timesteps, args.eval_freq, args.eval_episodes]
    if min(ints) <= 0 or min(args.curriculum_steps, args.stop_after_success_evals) < 0:
        ap.error("counts must be positive")
    torch.set_num_threads(args.torch_threads)

    normalize = args.normalize_observations or args.vecnormalize is not None
    if args.init and normalize and not args.vecnormalize:
        ap.error("--init with normalised observations requires --vecnormalize")

    hparams = dict(HPARAMS_SAC if args.algorithm == "SAC" else HPARAMS_PPO)
    if args.net_width:
        hparams["policy_kwargs"] = dict(net_arch=[args.net_width, args.net_width])
    if args.batch_size:
        hparams["batch_size"] = args.batch_size
    if args.learning_rate:
        hparams["learning_rate"] = args.learning_rate

    out = os.path.join("runs", args.run_name)
    if os.path.exists(os.path.join(out, "config.json")):
        ap.error(f"{out} already has a run; choose a new --run-name (earlier runs are never overwritten)")
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), "hyperparameters": hparams, "vecnormalize": normalize,
              "target_height": LIFT_HEIGHT, "origin": "continuation" if args.init else "from scratch"}
    Path(out, "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for src in ("env.py", "rewards.py", "observations.py", "callbacks.py"):
        Path(out, src).write_text(Path("so101_pick", src).read_text())
    Path(out, "train.py").write_text(Path(__file__).read_text())

    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(make_env(args.randomization), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.curriculum_steps > 0:
        venv.env_method("set_near_cube_start_probability", 1.0)
    # evaluation always starts from the normal pose (no curriculum) with the full goal
    eval_env = make_vec_env(make_env(args.randomization), n_envs=1, seed=args.seed + 1000)
    if normalize:
        venv = (VecNormalize.load(args.vecnormalize, venv) if args.vecnormalize
                else VecNormalize(venv, norm_obs=True, norm_reward=False))
        venv.training, venv.norm_reward = True, False
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
        # the eval env must see the training statistics: SB3's EvalCallback syncs them for us

    cls = {"SAC": SAC, "PPO": PPO}[args.algorithm]
    if args.init:
        model = cls.load(args.init, env=venv, tensorboard_log=out)
        if args.replay_buffer:
            model.load_replay_buffer(args.replay_buffer)
        print(f"CONTINUING from {args.init} (not a from-scratch run)")
    else:
        model = build_model(venv, out, args.algorithm, hparams)

    eval_cb = StageEvalCallback(
        eval_env, best_model_save_path=out, log_path=out, deterministic=True,
        callback_on_new_best=(SaveNormalizationCallback(Path(out, "vecnormalize_best.pkl")) if normalize else None),
        callback_after_eval=(StopOnFullSuccessCallback(args.stop_after_success_evals)
                             if args.stop_after_success_evals else None),
        eval_freq=max(args.eval_freq // args.n_envs, 1), n_eval_episodes=args.eval_episodes)
    callbacks = [eval_cb]
    if args.curriculum_steps > 0:
        callbacks.insert(0, ResetCurriculumCallback(args.curriculum_steps))

    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                    tb_log_name="tb", progress_bar=False, reset_num_timesteps=not bool(args.init))
    except KeyboardInterrupt:
        print("Interrupted; saving the current model.")
    finally:
        model.save(os.path.join(out, "final_model"))
        if normalize:
            venv.save(os.path.join(out, "vecnormalize_final.pkl"))
        venv.close()
        eval_env.close()
    print("Done. Models saved in", out)


if __name__ == "__main__":
    main()
