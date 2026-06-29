#!/usr/bin/env bash
set -e
cd /sessions/sharp-elegant-babbage/mnt/proof-of-deliberation
export PYTHONPATH=src
OUT=/tmp/rob
LOG=$OUT/progress.log
mkdir -p $OUT
echo "START $(date +%H:%M:%S)" > $LOG
for job in "synth sgd" "synth mlp" "gas sgd" "gas mlp"; do
  set -- $job; DS=$1; LR=$2
  echo ">>> $DS $LR begin $(date +%H:%M:%S)" >> $LOG
  python3 experiments/run_base_learner_robustness.py --dataset $DS --learner $LR --runs 10 --out $OUT >> $LOG 2>&1
  echo ">>> $DS $LR end   $(date +%H:%M:%S)" >> $LOG
done
echo "ALLDONE $(date +%H:%M:%S)" >> $LOG
