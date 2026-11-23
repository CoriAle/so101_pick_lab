# Lab: RL pick with the SO-101 arm (MuJoCo + Stable-Baselines3)

Teach a simulated SO-101 arm to **approach, grasp and lift** a 2.5 cm cube using
reinforcement learning. No ROS 2 needed: just Python, MuJoCo, Gymnasium and SB3.

**Setup:** follow `installation.md`, then run `python scripts/check_env.py`.

## What is provided vs. what you write

| File | Who | Purpose |
|---|---|---|
| `assets/so101/` | provided | SO-101 MuJoCo model (MuJoCo Menagerie, Apache-2.0) + `pick_scene.xml` (floor + cube) |
| `so101_pick/env.py` | provided | Gymnasium env: physics, actions, contacts, resets, success check |
| `so101_pick/rewards.py` | **YOU (TODO 1)** | the reward function |
| `so101_pick/observations.py` | provided | what the policy sees (21-D, documented in the file) |
| `train.py` | **YOU (TODO 2)** + provided harness | tune the baseline hyperparameters (SAC or PPO) |
| `evaluate.py` | provided | success rate over N episodes, optional video |
| `tests/` | provided | `test_env.py` (should pass), `test_reward.py` (checks your TODO 1) |

## The task in one paragraph
Each step the policy outputs 5 numbers in [-1, 1]: change of shoulder pan, shoulder lift, elbow,
wrist flex, and gripper (20 Hz, 150 steps per episode). The cube appears on the floor in front of
the arm. **Success** = both jaws touch the cube *and* the cube centre is >= 15 cm above its resting height (about 16.25 cm absolute Z).

## Workflow
1. `python scripts/check_env.py` and `python tests/test_env.py`: everything passes.
2. **Part A:** keep the starter reward and the baseline `HPARAMS` in `train.py`, and train briefly
   with `--randomization 0`. Look at TensorBoard: why does it not learn?
3. **Part B (TODO 1):** write the staged reward. Run `python tests/test_reward.py`, then train with
   `--randomization 0` (cube always in the same place).
4. **Part C (TODO 2):** tune the baseline hyperparameters (one change per run, different `--run-name`),
   then train with `--randomization 1` (random cube positions).
5. Evaluate: `python evaluate.py runs/<name>/best_model.zip --episodes 100 --video rollout.mp4`

```bash
python train.py --run-name my_run --timesteps 200000 --randomization 0
tensorboard --logdir runs            # Colab: %load_ext tensorboard ; %tensorboard --logdir runs
```

## Delivered result (this repo)
`entrega/` holds the delivered policy (`best_model.zip` + its `vecnormalize_best.pkl`), the rollout video and the
per-episode test evaluations. SAC trained from scratch in three stages (randomization 0 -> 0.3 -> 1.0), seed 0:

```bash
python train.py --run-name s1 --seed 0 --randomization 0 --timesteps 200000
python train.py --run-name s2 --seed 0 --randomization 0.3 --curriculum-steps 0 --timesteps 150000 \
    --init runs/s1/best_model.zip --vecnormalize runs/s1/vecnormalize_best.pkl
python train.py --run-name s3 --seed 0 --randomization 1.0 --curriculum-steps 0 --timesteps 400000 \
    --init runs/s2/final_model.zip --vecnormalize runs/s2/vecnormalize_final.pkl
python scripts/select_model.py runs/s3      # best checkpoint vs final model on 100 validation episodes (seed 500)
python evaluate.py runs/s3/best_model.zip --episodes 100 --video rollout.mp4
```

Test results (deterministic policy, simulation only): 100 % success on the fixed position (50 episodes) and
95.5 % on random positions (400 episodes, seeds 1 and 2). Failures are mostly unstable grasps that are lost
shortly after contact, and the jaws touch the floor in most episodes. Results depend on the seed.
Tests: `python -m pytest tests`.
