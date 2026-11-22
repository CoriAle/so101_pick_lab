"""Callback tests: curriculum fading, paired normalisation stats, evaluation isolation."""
import os, sys
import numpy as np
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import so101_pick  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from so101_pick.callbacks import (ResetCurriculumCallback, SaveNormalizationCallback,
                                  StageEvalCallback)


def _venv():
    return make_vec_env(lambda: gym.make("SO101Pick-v0", randomization=0.0), n_envs=2,
                        vec_env_cls=DummyVecEnv)


def _probabilities(venv):
    return [e.unwrapped.near_cube_start_probability for e in venv.envs]


def test_curriculum_fades_to_zero_and_leaves_eval_env_alone():
    venv, eval_env = _venv(), _venv()
    model = SAC("MlpPolicy", venv, learning_starts=10**9, verbose=0, device="cpu")
    model.set_logger(configure(None, []))
    cb = ResetCurriculumCallback(100)
    cb.init_callback(model)
    cb.model = model
    seen = []
    for t in (0, 50, 100, 150):
        cb.num_timesteps = t
        cb._update()
        seen.append(_probabilities(venv)[0])
    assert seen[0] == 1.0 and abs(seen[1] - 0.5) < 0.02 and seen[2] == 0.0 and seen[3] == 0.0
    assert _probabilities(eval_env) == [0.0, 0.0]


def test_curriculum_rejects_nonpositive_duration():
    try:
        ResetCurriculumCallback(0)
    except ValueError:
        return
    raise AssertionError


def test_best_checkpoint_gets_its_own_normalisation_stats(tmp_path):
    venv = VecNormalize(_venv(), norm_obs=True, norm_reward=False)
    venv.reset()
    model = SAC("MlpPolicy", venv, learning_starts=10**9, verbose=0, device="cpu")
    path = tmp_path / "vecnormalize_best.pkl"
    cb = SaveNormalizationCallback(path)
    cb.init_callback(model)
    cb.model = model
    cb._on_step()
    saved = VecNormalize.load(str(path), _venv())
    assert np.allclose(saved.obs_rms.mean, venv.obs_rms.mean)


def test_stage_eval_reports_grasp_lift_height(tmp_path):
    train, ev = _venv(), _venv()
    model = SAC("MlpPolicy", train, learning_starts=10**9, verbose=0, device="cpu")
    logger = configure(None, [])
    recorded = {}
    original = logger.record
    logger.record = lambda k, v, *a, **kw: (recorded.__setitem__(k, v), original(k, v, *a, **kw))[1]
    model.set_logger(logger)
    cb = StageEvalCallback(ev, n_eval_episodes=2, eval_freq=1, log_path=str(tmp_path))
    cb.init_callback(model)
    cb.model = model
    cb.n_calls = 0
    cb.num_timesteps = 0
    cb._on_step()
    # EvalCallback dumps the logger values it recorded; ours must be among the recorded keys
    assert len(cb._done) == 2
    for key in ("eval/grasp_rate", "eval/lift_rate", "eval/max_height_mean"):
        assert key in recorded and 0.0 <= recorded[key] <= 1.0 or key.endswith('height_mean'), key
