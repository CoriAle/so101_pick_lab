"""Tests for reward v2 (E5): stable grasp, grasp-loss cost, push cost, recovery signal."""
import copy, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv
from so101_pick.env import LIFT_HEIGHT
from so101_pick.rewards import compute_reward_v2 as R

ZERO = np.zeros(5, dtype=np.float32)


def base():
    e = SO101PickEnv(); e.reset(seed=0)
    return e._get_state()


def at(s, dist, lateral=0.0):
    s = copy.deepcopy(s)
    s["ee_pos"] = s["cube_pos"] + np.array([lateral, 0, dist], dtype=np.float32)
    s["ee_to_cube"] = s["cube_pos"] - s["ee_pos"]
    s["dist_ee_cube"] = float(np.hypot(dist, lateral))
    return s


def held(height, steps=3):
    s = at(base(), 0.005)
    s.update(fixed_jaw_contact=True, moving_jaw_contact=True, is_grasped=True, gripper_opening=0.05,
             cube_height=float(height), grasp_steps=steps, success=height >= LIFT_HEIGHT)
    return s


def test_v2_keeps_v1_ordering():
    s = base()
    assert R(at(s, 0.02), ZERO) > R(at(s, 0.10), ZERO)                 # closer is better
    assert R(held(0.0), ZERO) > R(at(s, 0.005), ZERO)                  # grasp beats hover
    assert R(held(0.05), ZERO) > R(held(0.0), ZERO)                    # lifting is better
    assert R(held(LIFT_HEIGHT), ZERO) > R(held(0.150), ZERO) + 0.5     # 15.1 cm is the goal
    air = held(0.05); air.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False)
    assert R(air, ZERO) < R(held(0.05), ZERO)                          # height without grasp


def test_stable_grasp_beats_momentary_touch():
    assert R(held(0.0, steps=3), ZERO) > R(held(0.0, steps=1), ZERO)


def test_losing_the_grasp_costs_reward():
    s = at(base(), 0.01); lost = copy.deepcopy(s); lost["grasp_lost"] = True
    assert R(lost, ZERO) < R(s, ZERO) - 0.2


def test_pushing_cube_far_costs_more_than_nudging():
    s = at(base(), 0.05)
    a, b = copy.deepcopy(s), copy.deepcopy(s)
    a["cube_xy_moved"], b["cube_xy_moved"] = 0.01, 0.10
    assert R(b, ZERO) < R(a, ZERO)


def test_far_cube_still_pulls_jaws_back():
    s = base()
    assert R(at(s, 0.0, lateral=0.20), ZERO) > R(at(s, 0.0, lateral=0.28), ZERO) + 0.01


def test_success_dominates_and_waiting_is_negative():
    assert R(held(0.15), ZERO) < 0
    assert R(held(LIFT_HEIGHT), ZERO) > 0


def test_env_tracks_grasp_steps_and_loss():
    e = SO101PickEnv(); e.reset(seed=0)
    s = e._get_state()
    assert s["grasp_steps"] == 0 and s["grasp_lost"] is False
    _, _, _, _, _ = e.step(np.zeros(5))
    assert e._get_state()["grasp_steps"] == 0


def test_env_counts_consecutive_grasp_and_flags_loss():
    e = SO101PickEnv(); e.reset(seed=0)
    real = e._get_state
    pattern = iter([True, True, True, False, False])

    def fake(last_action=None):
        s = real(last_action)
        s["is_grasped"] = next(pattern)
        return s
    e._get_state = fake
    seen = []
    captured = {}
    e.reward_fn = lambda st, a: captured.update(st) or 0.0
    for _ in range(5):
        e.step(np.zeros(5)); seen.append((captured["grasp_steps"], captured["grasp_lost"]))
    assert seen == [(1, False), (2, False), (3, False), (0, True), (0, False)]
