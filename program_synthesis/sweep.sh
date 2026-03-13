#!/usr/bin/env bash
# Run synthesize.py across 10 seeds on 2 GPUs in parallel.
# Usage: bash program_synthesis/sweep.sh [extra args...]
# Example: bash program_synthesis/sweep.sh --code-cells 15 --n-io 100 --program random_safe --n-test 1000 --constrained

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SEEDS=(1 2 3 4 5 6 7 8 9 10)
NGPUS=2
EXTRA_ARGS=("$@")

OUTDIR="program_synthesis/sweep_results"
mkdir -p "$OUTDIR"

pids=()

for seed in "${SEEDS[@]}"; do
    gpu=$(( (seed - 1) % NGPUS ))
    outfile="$OUTDIR/seed_${seed}.txt"
    echo "Launching seed=$seed on cuda:$gpu -> $outfile"
    CUDA_VISIBLE_DEVICES=$gpu python program_synthesis/synthesize.py \
        --seed "$seed" --device cuda "${EXTRA_ARGS[@]}" \
        > "$outfile" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "All done."

# Summary: extract final lines from each seed
echo ""
echo "===== SUMMARY ====="
for seed in "${SEEDS[@]}"; do
    outfile="$OUTDIR/seed_${seed}.txt"
    echo "--- seed $seed ---"
    # Print the instruction table and final accuracies
    sed -n '/^=====/,$ p' "$outfile"
    echo ""
done
