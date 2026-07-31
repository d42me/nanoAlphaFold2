#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/home/ubuntu/alphafold-decoded/nanoAlphaFold2
LEGACY_SOURCE=/tmp/af2-eval-dc8de3f/src

while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq runE-ablations; do
    sleep 60
done

cd "$PROJECT_ROOT"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate af2sprint
export PYTHONPATH="$LEGACY_SOURCE"

run_evaluation() {
    local evaluator=$1
    local run=$2
    local label=$3
    local checkpoint=$4
    local output="results/metrics/${run}_${evaluator}_${label}.csv"
    echo "[$(date -Is)] starting $run $evaluator $label"
    python "scripts/evaluate_${evaluator}.py" \
        --checkpoint "$checkpoint" \
        --output "$output" \
        2>&1 | tee "logs/${run}_${evaluator}_${label}.log"
}

run_evaluation cluster_paths run_f best checkpoints/ckpt_runF_no_profile_best.pt
run_evaluation cluster_paths run_f final checkpoints/ckpt_runF_no_profile.pt
run_evaluation cluster_covariance run_e best checkpoints/ckpt_runE_profile_dropout_best.pt
run_evaluation cluster_covariance run_e final checkpoints/ckpt_runE_profile_dropout.pt
run_evaluation cluster_covariance run_f best checkpoints/ckpt_runF_no_profile_best.pt
run_evaluation cluster_covariance run_f final checkpoints/ckpt_runF_no_profile.pt
