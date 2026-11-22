#!/usr/bin/env bash
# From-scratch chain identical to E1 -> E2 -> E3, varying only reward and seed:
#   s1: rand 0,   200k steps, mixture curriculum 50k   (from scratch)
#   s2: rand 0.3, 150k steps, from s1 best_model
#   s3: rand 1.0, 200k steps, from s2 final_model
# then evaluates s3 final: 50 fixed episodes + 2 x 200 random episodes (seeds 1, 2).
# Usage: scripts/run_chain.sh <v1|v2> <seed> <prefix>
set -euo pipefail
REWARD=$1; SEED=$2; P=$3
S1=${P}_s1_rand0; S2=${P}_s2_rand03; S3=${P}_s3_rand10
python train.py --run-name $S1 --reward $REWARD --seed $SEED --randomization 0 --timesteps 200000
python train.py --run-name $S2 --reward $REWARD --seed $SEED --randomization 0.3 --curriculum-steps 0 \
    --timesteps 150000 --init runs/$S1/best_model.zip --vecnormalize runs/$S1/vecnormalize_best.pkl
python train.py --run-name $S3 --reward $REWARD --seed $SEED --randomization 1.0 --curriculum-steps 0 \
    --timesteps 200000 --init runs/$S2/final_model.zip --vecnormalize runs/$S2/vecnormalize_final.pkl
M=runs/$S3/final_model.zip
python evaluate.py $M --episodes 50 --randomization 0 --json runs/$S3/eval_final_fixed.json
for E in 1 2; do
  python evaluate.py $M --episodes 200 --randomization 1 --seed $E --json runs/$S3/eval_final_random_seed$E.json
done
