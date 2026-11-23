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
from so101_pick.callbacks import ResetCurriculumCallback, SuccessSelectionEvalCallback


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


def test_best_model_and_its_normalisation_are_saved_together(tmp_path):
    venv = VecNormalize(_venv(), norm_obs=True, norm_reward=False)
    venv.reset()
    for _ in range(5):
        venv.step(np.zeros((2, 5)))
    model = SAC("MlpPolicy", venv, learning_starts=10**9, verbose=0, device="cpu")
    model.set_logger(configure(None, []))
    ev = VecNormalize(_venv(), training=False, norm_obs=True, norm_reward=False)
    cb = SuccessSelectionEvalCallback(ev, eval_freq=1, n_eval_episodes=1, save_dir=str(tmp_path))
    cb.init_callback(model)
    cb._evaluate = lambda: (1.0, 1.0, -10.0, 20)
    cb.n_calls = 1
    cb._on_step()
    assert (tmp_path / "best_model.zip").is_file() and (tmp_path / "evaluations_success.npz").is_file()
    saved = VecNormalize.load(str(tmp_path / "vecnormalize_best.pkl"), _venv())
    assert np.allclose(saved.obs_rms.mean, venv.obs_rms.mean)


def test_success_selection_prefers_success_over_reward(tmp_path):
    venv = _venv()
    model = SAC("MlpPolicy", venv, learning_starts=10**9, verbose=0, device="cpu")
    model.set_logger(configure(None, []))
    cb = SuccessSelectionEvalCallback(_venv(), eval_freq=1, n_eval_episodes=1, save_dir=str(tmp_path))
    cb.init_callback(model)
    results = iter([(0.9, 0.9, -5.0, 20), (1.0, 1.0, -30.0, 40), (0.95, 1.0, -1.0, 15), (1.0, 1.0, -20.0, 30)])
    cb._evaluate = lambda: next(results)
    saved = []
    model.save = lambda path: saved.append(cb.num_timesteps)
    for t in range(4):
        cb.n_calls, cb.num_timesteps = t + 1, t
        cb._on_step()
    # 0.9 -> best; 1.0 beats it despite lower reward; 0.95 with higher reward does NOT; 1.0 at -20 beats 1.0 at -30
    assert saved == [0, 1, 3]
    assert cb.best_key == (1.0, -20.0)


def test_success_selection_replays_identical_episodes(tmp_path):
    model = SAC("MlpPolicy", _venv(), learning_starts=10**9, verbose=0, device="cpu")
    model.set_logger(configure(None, []))
    ev = make_vec_env(lambda: gym.make("SO101Pick-v0", randomization=1.0), n_envs=1)
    cb = SuccessSelectionEvalCallback(ev, eval_freq=1, n_eval_episodes=1, save_dir=str(tmp_path), eval_seed=7)
    cb.init_callback(model)
    starts = []
    real_reset = ev.reset
    ev.reset = lambda: (starts.append(real_reset().copy()), starts[-1])[1]
    cb._evaluate(); cb._evaluate()
    assert np.allclose(starts[0], starts[1])
