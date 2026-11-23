#!/usr/bin/env bash
# Full recipe of the delivered model (from scratch, SAC, reward in so101_pick/rewards.py):
#   s1: randomization 0,   200k steps, mixture reset curriculum over 50k steps
#   s2: randomization 0.3, 150k steps, continues from s1 best_model
#   s3: randomization 1.0, 400k steps, continues from s2 final_model
# In every stage best_model = highest success on 20 identical eval episodes (eval seed 1000).
# Then: final selection between s3 best/final on 100 validation episodes (seed 500), and
# test evaluation of the selected model: 50 fixed + 2 x 200 random episodes (seeds 1, 2).
# Usage: scripts/run_chain.sh <seed> <prefix>
set -euo pipefail
SEED=$1; P=$2
S1=${P}_s1_rand0; S2=${P}_s2_rand03; S3=${P}_s3_rand10
python train.py --run-name $S1 --seed $SEED --randomization 0 --timesteps 200000
python train.py --run-name $S2 --seed $SEED --randomization 0.3 --curriculum-steps 0 \
    --timesteps 150000 --init runs/$S1/best_model.zip --vecnormalize runs/$S1/vecnormalize_best.pkl
python train.py --run-name $S3 --seed $SEED --randomization 1.0 --curriculum-steps 0 \
    --timesteps 400000 --init runs/$S2/final_model.zip --vecnormalize runs/$S2/vecnormalize_final.pkl
python scripts/select_model.py runs/$S3
M=runs/$S3/best_model.zip
python evaluate.py $M --episodes 50 --randomization 0 --json runs/$S3/test_fixed.json
for E in 1 2; do
  python evaluate.py $M --episodes 200 --randomization 1 --seed $E --json runs/$S3/test_random_seed$E.json
done
