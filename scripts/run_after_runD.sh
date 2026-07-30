#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/home/ubuntu/alphafold-decoded/nanoAlphaFold2
TRAINING_SESSION=runD

cd "$PROJECT_ROOT"
echo "waiting for tmux session $TRAINING_SESSION to finish"
while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$TRAINING_SESSION"; do
    sleep 60
done

if [[ $(tail -n 1 logs/train_runD_no_extra.log) != "done." ]]; then
    echo "Run D did not finish cleanly; follow-up evaluation cancelled"
    exit 1
fi

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate af2sprint

echo "Run D complete; starting main-MSA path evaluation"
PYTHONPATH=/tmp/af2-eval-dc8de3f/src \
    python scripts/evaluate_cluster_paths.py \
    --checkpoint checkpoints/ckpt_runD_no_extra.pt \
    --output results/metrics/run_d_cluster_paths.csv \
    2>&1 | tee logs/eval_runD_cluster_paths.log
