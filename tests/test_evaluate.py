"""evaluate.py helpers: terminal frame survives the auto-reset; paired normalisation is found."""
import os, sys
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import so101_pick  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv
from evaluate import TerminalFrameCapture, paired_normalization


def test_terminal_frame_is_captured_before_auto_reset():
    def make():
        return TerminalFrameCapture(gym.make("SO101Pick-v0", randomization=0.0,
                                             render_mode="rgb_array", max_episode_steps=3))
    v = DummyVecEnv([make]); v.reset()
    info = None
    for _ in range(3):
        _, _, done, infos = v.step([[0.0] * 5]); info = infos[0]
    assert done[0] and info["terminal_frame"].shape[-1] == 3


def test_paired_normalisation_matches_model_name(tmp_path):
    for n in ("best_model.zip", "final_model.zip", "vecnormalize_best.pkl"):
        (tmp_path / n).write_text("x")
    assert paired_normalization(str(tmp_path / "best_model.zip")).endswith("vecnormalize_best.pkl")
    assert paired_normalization(str(tmp_path / "final_model.zip")) is None   # no mismatched fallback
