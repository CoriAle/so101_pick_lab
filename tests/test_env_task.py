"""Task-specific env tests: 15.1 cm goal, both jaws, curriculum resets."""
import os, sys
import numpy as np
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import so101_pick  # noqa: F401
from so101_pick import SO101PickEnv
from so101_pick.env import CUBE_HALF, LIFT_HEIGHT


def _place_cube(env, height, grasp):
    """Put the cube `height` above rest and fake the jaw contacts through the state."""
    env.data.qpos[env._cube_qadr + 2] = CUBE_HALF + height
    import mujoco; mujoco.mj_forward(env.model, env.data)
    s = env._get_state()
    s["fixed_jaw_contact"], s["moving_jaw_contact"] = grasp
    s["is_grasped"] = all(grasp)
    s["success"] = bool(s["is_grasped"] and s["cube_height"] >= LIFT_HEIGHT)
    return s


def test_goal_is_15_1_cm_relative():
    assert LIFT_HEIGHT == 0.151


def test_success_needs_both_jaws_and_full_height():
    e = SO101PickEnv(); e.reset(seed=0)
    assert _place_cube(e, 0.151, (True, True))["success"]
    assert not _place_cube(e, 0.150, (True, True))["success"]      # 15.0 cm is not enough
    assert not _place_cube(e, 0.0, (True, True))["success"]        # grasp without lifting
    assert not _place_cube(e, 0.20, (True, False))["success"]      # lift with one jaw
    assert not _place_cube(e, 0.20, (False, False))["success"]


def test_real_contacts_do_not_fake_success_at_reset():
    e = SO101PickEnv(); _, info = e.reset(seed=0)
    assert not info["is_success"] and not info["is_grasped"]


def test_info_has_evaluation_fields():
    e = SO101PickEnv(); e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(5))
    assert {"arm_floor_contact", "gripper_opening", "is_success", "cube_height"} <= set(info)


def test_curriculum_reset_is_near_open_and_ungrasped():
    e = SO101PickEnv(near_cube_start_probability=1.0)
    for seed in range(5):
        _, info = e.reset(seed=seed)
        s = e._get_state()
        assert s["dist_ee_cube"] < 0.05, s["dist_ee_cube"]
        assert 0.2 < s["gripper_opening"] < 0.45
        assert not s["is_grasped"] and not s["success"]
        assert s["cube_height"] < 0.002                      # cube still on the floor


def test_curriculum_off_restores_standard_reset():
    e = SO101PickEnv(near_cube_start_probability=1.0); e.reset(seed=1)
    e.set_near_cube_start_probability(0.0)
    e.reset(seed=1)
    assert e._get_state()["dist_ee_cube"] > 0.05
    assert abs(e._get_state()["gripper_opening"] - 0.6) < 0.1


def test_curriculum_setter_validates_range():
    e = SO101PickEnv()
    for bad in (-0.1, 1.1):
        try:
            e.set_near_cube_start_probability(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_curriculum_reaches_env_through_gym_wrappers():
    from stable_baselines3.common.vec_env import DummyVecEnv
    v = DummyVecEnv([lambda: gym.make("SO101Pick-v0", randomization=0.0)])
    assert type(v.envs[0]) is not SO101PickEnv                 # really wrapped
    v.env_method("set_near_cube_start_probability", 1.0)
    assert v.envs[0].unwrapped.near_cube_start_probability == 1.0
    v.reset()
    assert v.envs[0].unwrapped._get_state()["dist_ee_cube"] < 0.05


def test_observation_contract_is_21d():
    e = SO101PickEnv(); o, _ = e.reset(seed=0)
    assert o.shape == (21,)
