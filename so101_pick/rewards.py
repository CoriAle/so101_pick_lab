"""The reward function (staged, bounded costs).

The task has three phases: APPROACH the cube, GRASP it, LIFT it.
A single sparse bonus on success (the original starter) gives the policy almost
no signal, so the reward is shaped per phase so that RL can discover the behaviour.

`state` is a dict (all SI units). Keys you can use:

    ee_pos            (3,)  position between the jaw tips
    cube_pos          (3,)  cube centre
    ee_to_cube        (3,)  cube_pos - ee_pos
    dist_ee_cube      float distance between jaw tips and cube centre (m)
    gripper_opening   float jaw joint angle (rad): ~-0.17 closed ... 1.75 wide open
                            (the cube is 2.5 cm wide: ~0.3 rad is "just wider than the cube")
    cube_height       float cube centre height above its resting height (m); 0 on the floor
    cube_xy_moved     float how far the cube slid sideways since reset (m)
    fixed_jaw_contact bool  fixed jaw touches the cube
    moving_jaw_contact bool moving jaw touches the cube
    is_grasped        bool  BOTH jaws touch the cube
    arm_floor_contact bool  some part of the robot touches the floor
    joint_vel         (5,)  joint velocities (rad/s)
    success           bool  grasped AND cube_height >= 15.1 cm (episode ends)
    last_action       (5,)  the action just applied, in [-1, 1]
    step              int   steps taken so far in this episode: 0 at reset, then 1, 2, ...
                            An episode is at most 150 steps (20 Hz = 7.5 s), so
                            `150 - state["step"]` is the number of steps left.
                            Use it to encourage quickness (see the design questions).

`action` is the same 5-vector in [-1, 1].

Design questions to think about:
  * Why is a reward based on distance (dense) easier to learn from than a
    single success bonus (sparse)?
  * How do you make the reward "switch on" the next phase only when the
    previous one is done? (e.g. only reward lifting once the cube is grasped.)
  * Can the policy cheat? Think about: knocking the cube away, hovering next to
    the cube forever, slamming the arm into the floor, jittery actions.
  * Scale matters: keep each term at a similar magnitude (roughly <= 1 per step),
    and make success clearly the best outcome.
  * Quickness: success ends the episode. How can you make finishing sooner better
    than finishing later? (e.g. a small cost per step, or a success bonus that grows
    with the number of steps left.) Beware: if the per-step rewards you collect while
    holding the cube add up to more than the success bonus, the policy can learn to
    stall just below the success height.

Check your work with:  python -m pytest tests/test_reward.py   (or: python tests/test_reward.py)
"""
import numpy as np

from .env import LIFT_HEIGHT

MAX_STEPS = 150

# Reward = bounded costs (tanh) per phase; initial weights/scales; see entrega/REPORT.md.
# Every cost lives in [0, weight], so no single term can dominate the success bonus.


def _approach_cost(lateral: float, dz: float) -> float:
    """Lateral alignment and descent are separate, so hovering above the cube still costs."""
    return 0.35 * np.tanh(lateral / 0.04) + 0.55 * np.tanh(abs(dz) / 0.025)


def _close_readiness(dist: float, lateral: float) -> float:
    """1 when the jaws are around the cube and aligned, falling to 0 as they drift away."""
    near = 1.0 - np.tanh(max(0.0, dist - 0.008) / 0.01)
    aligned = 1.0 - np.tanh(max(0.0, lateral - 0.005) / 0.01)
    return near * aligned


def _opening_cost(opening: float, ready: float) -> float:
    premature = np.tanh(max(0.0, 0.3 - opening) / 0.2)    # closed too early
    missed = np.tanh(max(0.0, opening - 0.05) / 0.6)      # still open when it should close
    excess = np.tanh(max(0.0, opening - 0.6) / 0.6)       # needlessly wide
    return 0.2 * (1.0 - ready) * premature + 0.3 * ready * missed + 0.2 * excess


def compute_reward(state: dict, action: np.ndarray) -> float:
    grasped = bool(state["is_grasped"])
    offset = np.asarray(state["ee_to_cube"])
    lateral = float(np.linalg.norm(offset[:2]))
    ready = _close_readiness(state["dist_ee_cube"], lateral)

    touching = bool(state["fixed_jaw_contact"] or state["moving_jaw_contact"])
    # one jaw touching earns a small credit, both jaws (grasp) remove the grasp cost entirely
    grasp_cost = 0.0 if grasped else (
        0.3 + _opening_cost(state["gripper_opening"], ready) - 0.12 * ready * touching)

    # lift progress counts only while the cube is held
    progress = 0.0
    if grasped:
        fraction = np.clip(state["cube_height"] / LIFT_HEIGHT, 0.0, 1.0)
        progress = float(np.tanh(fraction) / np.tanh(1.0))
    lift_cost = 0.4 * (1.0 - progress)

    slide = 0.1 * np.tanh(max(0.0, state["cube_xy_moved"]) / 0.05)
    if grasped:                       # carrying the cube upward is not "sliding"
        slide *= 1.0 - np.tanh(max(0.0, state["cube_height"]) / 0.025)
    floor = 0.5 * bool(state["arm_floor_contact"])
    effort = 0.01 * np.mean(np.square(np.clip(action, -1.0, 1.0)))
    speed = 0.01 * np.mean(np.square(np.tanh(np.asarray(state["joint_vel"]) / 2.0)))

    reward = -0.01 - slide - floor - effort - speed
    if state["success"]:
        # finishing sooner is worth more: bonus between 1 and 2
        reward += 1.0 + float(np.clip((MAX_STEPS - state["step"]) / MAX_STEPS, 0.0, 1.0))
    else:
        reward -= _approach_cost(lateral, float(offset[2])) + grasp_cost + lift_cost
    return float(reward)

