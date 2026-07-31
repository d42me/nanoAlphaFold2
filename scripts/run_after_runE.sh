#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/home/ubuntu/alphafold-decoded/nanoAlphaFold2
LEGACY_SOURCE=/tmp/af2-eval-dc8de3f/src

cd "$PROJECT_ROOT"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate af2sprint
export PYTHONPATH="$LEGACY_SOURCE"

run_factorial() {
    local label=$1
    local checkpoint=$2
    echo "[$(date -Is)] starting Run E $label factorial"
    python scripts/evaluate_cluster_paths.py \
        --checkpoint "$checkpoint" \
        --output "results/metrics/run_e_cluster_paths_${label}.csv" \
        2>&1 | tee "logs/run_e_cluster_paths_${label}.log"
}

run_factorial best checkpoints/ckpt_runE_profile_dropout_best.pt
run_factorial final checkpoints/ckpt_runE_profile_dropout.pt

echo "[$(date -Is)] starting Run F"
python scripts/train_multi.py \
    --tag ckpt_runF_no_profile \
    --big \
    --steps 250000 \
    --cosine \
    --min-lr 1e-4 \
    --exclude-file configs/splits/exclude_runC.txt \
    --no-extra-msa \
    --profile-dropout 1.0 \
    2>&1 | tee logs/train_runF_no_profile.log
