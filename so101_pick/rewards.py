"""TODO 1: the reward function  (this is the heart of the lab).

The task has three phases: APPROACH the cube, GRASP it, LIFT it.
With the starter reward below (a single sparse bonus on success), the policy
almost never sees any reward, so it learns nothing. Your job is to shape the
reward so that RL can discover the whole behaviour.

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
    success           bool  grasped AND cube_height >= 5 cm (episode ends)
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


def compute_reward(state: dict, action: np.ndarray) -> float:
    dist = float(state["dist_ee_cube"])
    grasped = bool(state.get("is_grasped", False))
    height = max(0.0, float(state.get("cube_height", 0.0)))
    success = bool(state.get("success", False))
    previous_dist = float(state.get("prev_dist_ee_cube", dist))
    previous_height = max(0.0, float(state.get("prev_cube_height", height)))

    if not grasped:
        # Phase 1: Approach & prepare gripper
        # Use progress as the main signal so a policy cannot profit by
        # stopping at the nearest rewarding distance.
        r_approach = 10.0 * (previous_dist - dist)
        r_approach += 0.05 * (1.0 - np.tanh(5.0 * dist))
        opening = float(state.get("gripper_opening", 0.6))
        # Keep gripper open while flying to cube, close only in grasp range.
        if dist > 0.03:
            r_grip = 0.05 * float(np.clip(opening / 0.6, 0.0, 1.0))
        else:
            r_grip = 0.2 * float(np.clip((0.6 - opening) / 0.6, 0.0, 1.0))
        r_contact = 0.5 if state.get("fixed_jaw_contact", False) else 0.0
        r_contact += 0.5 if state.get("moving_jaw_contact", False) else 0.0
        r_lift = 0.0
    else:
        # Phase 2: Grasped! Now the entire objective is to LIFT.
        r_approach = 0.0
        r_grip = 0.4
        r_contact = 1.3
        r_lift = 30.0 * (height - previous_height)

    # Success bonus: clearly dominate every non-success reward.
    r_success = 0.0
    if success:
        r_success = 100.0

    # Penalties
    r_floor = -0.5 if state.get("arm_floor_contact", False) else 0.0
    r_slide = -0.5 * float(state.get("cube_xy_moved", 0.0)) if not grasped else 0.0
    r_action = -0.01 * float(np.sum(np.square(action)))
    r_vel = -0.002 * float(np.sum(np.square(state.get("joint_vel", 0.0))))
    prev_action = np.asarray(state.get("prev_action", np.zeros(5)), dtype=np.float32)
    r_jerk = -0.02 * float(np.sum(np.square(action - prev_action)))
    r_time = -0.02

    reward = (
        r_approach
        + r_grip
        + r_contact
        + r_lift
        + r_success
        + r_floor
        + r_slide
        + r_action
        + r_vel
        + r_jerk
        + r_time
    )
    return float(reward)
