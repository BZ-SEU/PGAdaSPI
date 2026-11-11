#!/usr/bin/env bash
# Split an image list round-robin over GPUs; one scripts/run.py process per GPU, logs next to the output.
# Usage: scripts/run_parallel.sh "<gpu ids>" "<image ids>" <output dir> [run.py args...]
#   scripts/run_parallel.sh "0 1 2 3" "Monarch Parrots barbara" outputs/abl_full \
#       --template SET11_256 --dataset-root data/set11_proxunroll/native --ablation full
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/baojr/.conda/envs/PIE2E/bin/python}
read -r -a gpus <<< "$1"; read -r -a ids <<< "$2"; out=$3; shift 3
mkdir -p "$out"
for g in "${!gpus[@]}"; do
  mine=()
  for i in "${!ids[@]}"; do (( i % ${#gpus[@]} == g )) && mine+=("${ids[$i]}"); done
  (( ${#mine[@]} )) || continue
  "$PY" scripts/run.py "$@" --image-ids "${mine[@]}" --device "cuda:${gpus[$g]}" --output "$out" \
    > "$out/gpu${gpus[$g]}.log" 2>&1 &
done
wait
"$PY" scripts/summarize.py "$out" > "$out/summary.json"
echo "done: $out/summary.json"
