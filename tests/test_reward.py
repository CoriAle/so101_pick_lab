"""Self-check for TODO 1 (your reward). These tests use synthetic states, no training needed.
They FAIL with the starter reward and should PASS once your staged reward is sensible.
    python -m pytest tests/test_reward.py -v      (or)      python tests/test_reward.py
Passing is necessary, not sufficient: a good reward must still *train*.
"""
import copy, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv
from so101_pick.env import LIFT_HEIGHT
from so101_pick.rewards import compute_reward

ZERO = np.zeros(5, dtype=np.float32)


def base_state():
    env = SO101PickEnv(); env.reset(seed=0)
    return env._get_state()


def with_dist(s, dist):
    """Place the jaws `dist` metres straight above the cube."""
    s = copy.deepcopy(s)
    s["ee_pos"] = s["cube_pos"] + np.array([0, 0, dist], dtype=np.float32)
    s["ee_to_cube"] = s["cube_pos"] - s["ee_pos"]
    s["dist_ee_cube"] = float(dist)
    return s


def grasped(s, height):
    s = with_dist(s, 0.005)
    s.update(fixed_jaw_contact=True, moving_jaw_contact=True, is_grasped=True,
             cube_height=float(height), gripper_opening=0.05)
    s["cube_pos"] = s["cube_pos"].copy(); s["cube_pos"][2] = 0.0125 + height
    s["success"] = height >= LIFT_HEIGHT
    return s


def test_reward_is_finite_float():
    r = compute_reward(base_state(), ZERO)
    assert np.isfinite(r)


def test_getting_closer_is_rewarded():
    s = base_state()
    assert compute_reward(with_dist(s, 0.02), ZERO) > compute_reward(with_dist(s, 0.10), ZERO)


def test_grasping_beats_hovering():
    s = base_state()
    assert compute_reward(grasped(s, 0.0), ZERO) > compute_reward(with_dist(s, 0.005), ZERO)


def test_lifting_while_grasped_is_rewarded():
    s = base_state()
    assert compute_reward(grasped(s, 0.03), ZERO) > compute_reward(grasped(s, 0.0), ZERO)


def test_success_is_the_best_outcome():
    s = base_state()
    best_non_success = compute_reward(grasped(s, 0.149), ZERO)
    assert compute_reward(grasped(s, LIFT_HEIGHT), ZERO) > best_non_success


def test_closing_gripper_in_empty_air_is_not_rewarded():
    s = with_dist(base_state(), 0.10)
    closed = copy.deepcopy(s); closed["gripper_opening"] = -0.1
    opened = copy.deepcopy(s); opened["gripper_opening"] = 0.6
    assert compute_reward(closed, ZERO) <= compute_reward(opened, ZERO) + 1e-6


def test_lifting_without_grasp_is_not_rewarded():
    """Cube 3 cm up but NOT grasped (e.g. thrown/balanced) must not look like progress."""
    s = base_state()
    g = grasped(s, 0.03)
    air = copy.deepcopy(g); air.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False)
    assert compute_reward(air, ZERO) < compute_reward(g, ZERO)


def test_14_9cm_is_not_success_but_15cm_is():
    s = base_state()
    assert not grasped(s, 0.149)["success"] and grasped(s, 0.150)["success"]
    assert compute_reward(grasped(s, 0.150), ZERO) > compute_reward(grasped(s, 0.149), ZERO) + 0.5


def test_earlier_success_is_better():
    s = grasped(base_state(), LIFT_HEIGHT)
    early, late = copy.deepcopy(s), copy.deepcopy(s)
    early["step"], late["step"] = 20, 140
    assert compute_reward(early, ZERO) > compute_reward(late, ZERO)


def test_waiting_below_success_height_costs_reward():
    s = base_state()
    assert compute_reward(grasped(s, 0.149), ZERO) < 0   # every non-success step is net negative


def test_one_jaw_contact_beats_none_but_not_full_grasp():
    s = with_dist(base_state(), 0.005); s["gripper_opening"] = 0.3
    one = copy.deepcopy(s); one["fixed_jaw_contact"] = True
    assert compute_reward(one, ZERO) > compute_reward(s, ZERO)
    assert compute_reward(grasped(base_state(), 0.0), ZERO) > compute_reward(one, ZERO)


def test_descending_beats_hovering_with_open_jaws():
    s = base_state()
    hover, low = with_dist(s, 0.08), with_dist(s, 0.02)
    for x in (hover, low): x["gripper_opening"] = 0.5
    assert compute_reward(low, ZERO) > compute_reward(hover, ZERO)


def test_excessive_opening_is_worse_than_capture_opening():
    s = with_dist(base_state(), 0.05)
    wide, capture = copy.deepcopy(s), copy.deepcopy(s)
    wide["gripper_opening"], capture["gripper_opening"] = 1.6, 0.5
    assert compute_reward(capture, ZERO) > compute_reward(wide, ZERO)


def test_closing_is_encouraged_when_aligned_at_cube():
    s = with_dist(base_state(), 0.0)
    closing, open_ = copy.deepcopy(s), copy.deepcopy(s)
    closing["gripper_opening"], open_["gripper_opening"] = 0.1, 0.6
    assert compute_reward(closing, ZERO) > compute_reward(open_, ZERO)


def test_floor_sliding_and_excessive_motion_are_penalized():
    s = with_dist(base_state(), 0.05)
    base = compute_reward(s, ZERO)
    for key, val in (("arm_floor_contact", True), ("cube_xy_moved", 0.1)):
        x = copy.deepcopy(s); x[key] = val
        assert compute_reward(x, ZERO) < base
    assert compute_reward(s, np.ones(5, dtype=np.float32)) < base
    fast = copy.deepcopy(s); fast["joint_vel"] = np.full(5, 5.0, dtype=np.float32)
    assert compute_reward(fast, ZERO) < base


def test_dropping_the_cube_is_worse():
    s = base_state()
    assert compute_reward(grasped(s, 0.10), ZERO) > compute_reward(grasped(s, 0.02), ZERO)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, t in tests:
        try:
            t(); print("PASS", name)
        except AssertionError:
            failed += 1; print("FAIL", name)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
